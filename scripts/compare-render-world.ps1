#requires -Version 7.2

<#
.SYNOPSIS
    Compares two immutable Myth Engine revisions with helmet_gltf_fixed_probe.

.DESCRIPTION
    The runner never checks out either revision. It exports each commit with
    `git archive` into target/perf-snapshots, builds the snapshots into separate
    Cargo target directories, and then runs one GPU process at a time.

    Every arm uses alternating ABBA/BAAB blocks. A block therefore contributes
    two raw samples per revision and one drift-resistant paired log-ratio. Raw
    process results are written to CSV after every successful run.

    Timing builds always use --locked and --no-default-features. The default
    explicit feature set is identical to the workspace defaults, but does not
    enable rdg_bench_support because its instrumentation changes the endpoint.

.EXAMPLE
    pwsh scripts/compare-render-world.ps1

.EXAMPLE
    pwsh scripts/compare-render-world.ps1 `
        -BaselineRef 209daedf -CandidateRef HEAD `
        -WarmupFrames 10000 -MeasureFrames 200000 -Blocks 8

.EXAMPLE
    pwsh scripts/compare-render-world.ps1 -PlanOnly -Blocks 2

.EXAMPLE
    pwsh scripts/compare-render-world.ps1 -AnalyzeOnly `
        -OutputDirectory target/perf-results/render-world-existing
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$BaselineRef = '209daedf',

    [ValidateNotNullOrEmpty()]
    [string]$CandidateRef = 'HEAD',

    [ValidateRange(0, 100000000)]
    [int]$WarmupFrames = 10000,

    [ValidateRange(1, 100000000)]
    [int]$MeasureFrames = 200000,

    [ValidateRange(1, 1000)]
    [int]$Blocks = 8,

    [ValidateNotNullOrEmpty()]
    [string]$Arms = 'FullOff,NoDraw,Untextured,FullAuto',

    [ValidateNotNullOrEmpty()]
    [string]$Features = 'winit,gltf,http,advanced_noise',

    [ValidateSet('dx12', 'vulkan', 'metal', 'gl', 'webgpu')]
    [string]$Backend = 'dx12',

    [string]$ExpectedAdapter = '',

    [ValidateRange(1, 86400)]
    [int]$ProcessTimeoutSeconds = 1800,

    [ValidateRange(1000, 1000000)]
    [int]$BootstrapResamples = 20000,

    [uint32]$BootstrapSeed = 1297699912,

    [string]$OutputDirectory = '',

    [switch]$AnalyzeOnly,

    [switch]$PlanOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$InvariantCulture = [Globalization.CultureInfo]::InvariantCulture
$ProbeRelativePath = 'examples/helmet_gltf_fixed_probe.rs'
$ProbeResultPrefix = 'HELMET_FIXED_PROBE '
$ProbeEnvironmentNames = @(
    'MYTH_HELMET_PROBE_WARMUP_FRAMES',
    'MYTH_HELMET_PROBE_MEASURE_FRAMES',
    'MYTH_HELMET_PROBE_DISABLE_ENV',
    'MYTH_HELMET_PROBE_UNTEXTURED',
    'MYTH_HELMET_PROBE_NO_DRAW',
    'MYTH_HELMET_PROBE_GPU_DRIVEN'
)
$AllowedLocalOnlyStatusEntries = @(
    '?? docs/articles/rdg-deferred-prepare-refactor.md'
)

function Get-NativeCommandPath {
    param([Parameter(Mandatory)][string]$Name)

    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -eq $command) {
        throw "Required command '$Name' was not found."
    }
    return $command.Source
}

function Convert-BackendName {
    param([Parameter(Mandatory)][string]$Name)

    $normalized = ($Name -replace '[^A-Za-z0-9]', '').ToLowerInvariant()
    switch ($normalized) {
        'browserwebgpu' { return 'webgpu' }
        'opengl' { return 'gl' }
        'gles' { return 'gl' }
        default { return $normalized }
    }
}

function Get-HostEnvironmentSnapshot {
    $videoControllers = @()
    $powerScheme = ''
    if ($IsWindows) {
        try {
            $videoControllers = @(
                Get-CimInstance Win32_VideoController -ErrorAction Stop |
                    Sort-Object PNPDeviceID |
                    Select-Object Name, DriverVersion, PNPDeviceID,
                        CurrentHorizontalResolution, CurrentVerticalResolution
            )
        } catch {
            $videoControllers = @([pscustomobject]@{ Error = $_.Exception.Message })
        }
        $powercfg = Get-Command powercfg.exe -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $powercfg) {
            try {
                $powerScheme = ((& $powercfg.Source /getactivescheme 2>&1) -join ' ').Trim()
            } catch {
                $powerScheme = "unavailable: $($_.Exception.Message)"
            }
        }
    }

    return [pscustomobject][ordered]@{
        MachineName = [Environment]::MachineName
        OsDescription = [Runtime.InteropServices.RuntimeInformation]::OSDescription
        OsArchitecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        ProcessArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
        ProcessorIdentifier = [Environment]::GetEnvironmentVariable('PROCESSOR_IDENTIFIER')
        LogicalProcessorCount = [Environment]::ProcessorCount
        PowerScheme = $powerScheme
        VideoControllers = $videoControllers
    }
}

function Resolve-FullPath {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$BasePath
    )

    if ([IO.Path]::IsPathFullyQualified($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Assert-SafeChildPath {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root
    )

    $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $fullRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to mutate path outside $fullRoot`: $fullPath"
    }
}

function Invoke-GitText {
    param([Parameter(Mandatory)][string[]]$Arguments)

    $output = & $script:GitPath -C $script:RepoRoot @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed:`n$($output -join [Environment]::NewLine)"
    }
    return @($output | ForEach-Object { $_.ToString() })
}

function Resolve-Revision {
    param(
        [Parameter(Mandatory)][ValidateSet('baseline', 'candidate')][string]$Role,
        [Parameter(Mandatory)][string]$Ref
    )

    $revisionOutput = Invoke-GitText @('rev-parse', '--verify', "$Ref^{commit}")
    $oid = $revisionOutput |
        Where-Object { $_ -match '^[0-9a-fA-F]{40}$' } |
        Select-Object -Last 1
    if ($null -eq $oid) {
        throw "Could not resolve '$Ref' to a commit object."
    }
    $oid = $oid.ToLowerInvariant()

    & $script:GitPath -C $script:RepoRoot cat-file -e "${oid}:$script:ProbeRelativePath"
    if ($LASTEXITCODE -ne 0) {
        throw "Revision '$Ref' ($oid) does not contain $script:ProbeRelativePath."
    }

    [pscustomobject]@{
        Role = $Role
        Ref = $Ref
        Oid = $oid
        ShortOid = $oid.Substring(0, 12)
        SnapshotPath = ''
        TargetPath = ''
        BinaryPath = ''
    }
}

function Get-ArmConfiguration {
    param([Parameter(Mandatory)][string]$Arm)

    switch ($Arm.ToLowerInvariant()) {
        'fulloff' {
            return [pscustomobject]@{
                Name = 'FullOff'
                DisableEnvironment = $false
                Untextured = $false
                NoDraw = $false
                GpuDriven = 'off'
            }
        }
        'nodraw' {
            return [pscustomobject]@{
                Name = 'NoDraw'
                DisableEnvironment = $true
                Untextured = $false
                NoDraw = $true
                GpuDriven = 'off'
            }
        }
        'untextured' {
            return [pscustomobject]@{
                Name = 'Untextured'
                DisableEnvironment = $true
                Untextured = $true
                NoDraw = $false
                GpuDriven = 'off'
            }
        }
        'fullauto' {
            return [pscustomobject]@{
                Name = 'FullAuto'
                DisableEnvironment = $false
                Untextured = $false
                NoDraw = $false
                GpuDriven = 'auto'
            }
        }
        default { throw "Unknown arm '$Arm'." }
    }
}

function Get-BlockPattern {
    param([Parameter(Mandatory)][int]$Block)

    if (($Block % 2) -eq 1) {
        return @('baseline', 'candidate', 'candidate', 'baseline')
    }
    return @('candidate', 'baseline', 'baseline', 'candidate')
}

function Convert-ProbeResultLine {
    param([Parameter(Mandatory)][string]$Line)

    if (-not $Line.StartsWith($script:ProbeResultPrefix, [StringComparison]::Ordinal)) {
        throw "Not a helmet fixed-probe result line: $Line"
    }

    $fields = @{}
    $payload = $Line.Substring($script:ProbeResultPrefix.Length)
    foreach ($token in ($payload -split '\s+')) {
        if ($token -notmatch '^(?<key>[A-Za-z0-9_]+)=(?<value>.+)$') {
            throw "Malformed fixed-probe token '$token'."
        }
        $fields[$Matches.key] = $Matches.value
    }

    $required = @(
        'warmup', 'frames', 'environment', 'textured', 'draw_model', 'gpu_driven',
        'total_ms', 'mean_us', 'median_us', 'p95_us', 'p99_us', 'max_us',
        'mean_fps', 'median_fps', 'p95_fps', 'p99_fps'
    )
    foreach ($name in $required) {
        if (-not $fields.ContainsKey($name)) {
            throw "Fixed-probe result is missing '$name': $Line"
        }
    }

    return [pscustomobject]@{
        Warmup = [int]::Parse($fields.warmup, $script:InvariantCulture)
        Frames = [int]::Parse($fields.frames, $script:InvariantCulture)
        Environment = [bool]::Parse($fields.environment)
        Textured = [bool]::Parse($fields.textured)
        DrawModel = [bool]::Parse($fields.draw_model)
        GpuDriven = $fields.gpu_driven
        TotalMs = [double]::Parse($fields.total_ms, $script:InvariantCulture)
        MeanUs = [double]::Parse($fields.mean_us, $script:InvariantCulture)
        MedianUs = [double]::Parse($fields.median_us, $script:InvariantCulture)
        P95Us = [double]::Parse($fields.p95_us, $script:InvariantCulture)
        P99Us = [double]::Parse($fields.p99_us, $script:InvariantCulture)
        MaxUs = [double]::Parse($fields.max_us, $script:InvariantCulture)
        MeanFps = [double]::Parse($fields.mean_fps, $script:InvariantCulture)
        MedianFps = [double]::Parse($fields.median_fps, $script:InvariantCulture)
        P95Fps = [double]::Parse($fields.p95_fps, $script:InvariantCulture)
        P99Fps = [double]::Parse($fields.p99_fps, $script:InvariantCulture)
    }
}

function Assert-ProbeParser {
    $fixture = 'HELMET_FIXED_PROBE warmup=0 frames=1 environment=true textured=false draw_model=true gpu_driven=Off total_ms=1.250 mean_us=1250.000 median_us=1250.000 p95_us=1250.000 p99_us=1250.000 max_us=1250.000 mean_fps=800.00 median_fps=800.00 p95_fps=800.00 p99_fps=800.00'
    $parsed = Convert-ProbeResultLine $fixture
    if ($parsed.Frames -ne 1 -or $parsed.TotalMs -ne 1.25 -or $parsed.Textured) {
        throw 'Internal fixed-probe parser smoke test failed.'
    }
}

function Get-LinearPercentile {
    param(
        [Parameter(Mandatory)][double[]]$SortedValues,
        [Parameter(Mandatory)][ValidateRange(0.0, 1.0)][double]$Probability
    )

    if ($SortedValues.Count -eq 0) {
        throw 'A percentile requires at least one value.'
    }
    if ($SortedValues.Count -eq 1) {
        return $SortedValues[0]
    }
    $rank = ($SortedValues.Count - 1) * $Probability
    $lower = [int][Math]::Floor($rank)
    $upper = [int][Math]::Ceiling($rank)
    if ($lower -eq $upper) {
        return $SortedValues[$lower]
    }
    $weight = $rank - $lower
    return $SortedValues[$lower] * (1.0 - $weight) + $SortedValues[$upper] * $weight
}

function Get-PairedLogBootstrapCi {
    param(
        [Parameter(Mandatory)][double[]]$LogRatios,
        [Parameter(Mandatory)][int]$Resamples,
        [Parameter(Mandatory)][uint32]$Seed
    )

    if ($LogRatios.Count -eq 0) {
        throw 'A paired bootstrap requires at least one block.'
    }

    # A local 32-bit LCG makes the resampling stream independent of changes to
    # System.Random across .NET versions. Sampling units are whole ABBA/BAAB
    # blocks, preserving the paired experimental design.
    $modulus = [uint64]4294967296
    $state = [uint64]$Seed
    $means = [double[]]::new($Resamples)
    for ($sample = 0; $sample -lt $Resamples; $sample++) {
        $sum = 0.0
        for ($draw = 0; $draw -lt $LogRatios.Count; $draw++) {
            $state = ([uint64]1664525 * $state + [uint64]1013904223) % $modulus
            # Do not use the low LCG bits with modulo: for 2/4/8 blocks their
            # short period would make every resample contain the same strata
            # and collapse the confidence interval. Multiply-high maps the
            # full 32-bit state into [0, block_count) without that artifact.
            $scaled = $state * [uint64]$LogRatios.Count
            $index = [int]($scaled -shr 32)
            $sum += $LogRatios[$index]
        }
        $means[$sample] = $sum / $LogRatios.Count
    }
    [Array]::Sort($means)

    return [pscustomobject]@{
        LowerLogRatio = Get-LinearPercentile -SortedValues $means -Probability 0.025
        UpperLogRatio = Get-LinearPercentile -SortedValues $means -Probability 0.975
    }
}

function Get-FormalGateVerdict {
    param(
        [Parameter(Mandatory)][double]$MeanCiLowerPercent,
        [Parameter(Mandatory)][double]$MeanCiUpperPercent,
        [Parameter(Mandatory)][double]$PairedP95DeltaPercent,
        [Parameter(Mandatory)][int]$PairedBlocks,
        [Parameter(Mandatory)][int]$WarmupFrameCount,
        [Parameter(Mandatory)][int]$MeasureFrameCount
    )

    $meanPassed = $MeanCiLowerPercent -ge -2.0 -and $MeanCiUpperPercent -le 2.0
    $p95Passed = $PairedP95DeltaPercent -lt 5.0
    $eligible = $PairedBlocks -ge 8 -and
        $WarmupFrameCount -ge 10000 -and
        $MeasureFrameCount -ge 200000
    $verdict = if (-not $eligible) {
        'INSUFFICIENT_SAMPLES'
    } elseif ($meanPassed -and $p95Passed) {
        'PASS'
    } else {
        'FAIL'
    }
    return [pscustomobject]@{
        MeanPassed = $meanPassed
        P95Passed = $p95Passed
        Eligible = $eligible
        Verdict = $verdict
    }
}

function Assert-StatisticsHelpers {
    $fixture = [double[]]@(-0.02, -0.005, 0.01, 0.025)
    $first = Get-PairedLogBootstrapCi -LogRatios $fixture -Resamples 1000 -Seed 12345
    $second = Get-PairedLogBootstrapCi -LogRatios $fixture -Resamples 1000 -Seed 12345
    if (
        $first.LowerLogRatio -ne $second.LowerLogRatio -or
        $first.UpperLogRatio -ne $second.UpperLogRatio -or
        $first.LowerLogRatio -ge $first.UpperLogRatio
    ) {
        throw 'Internal deterministic bootstrap smoke test failed.'
    }
    $percentileFixture = [double[]]@(0.0, 10.0)
    if ((Get-LinearPercentile $percentileFixture 0.25) -ne 2.5) {
        throw 'Internal percentile smoke test failed.'
    }
    $passingGate = Get-FormalGateVerdict -MeanCiLowerPercent -1.0 `
        -MeanCiUpperPercent 1.0 -PairedP95DeltaPercent 4.999 -PairedBlocks 8 `
        -WarmupFrameCount 10000 -MeasureFrameCount 200000
    $p95Boundary = Get-FormalGateVerdict -MeanCiLowerPercent -1.0 `
        -MeanCiUpperPercent 1.0 -PairedP95DeltaPercent 5.0 -PairedBlocks 8 `
        -WarmupFrameCount 10000 -MeasureFrameCount 200000
    $shortRun = Get-FormalGateVerdict -MeanCiLowerPercent -1.0 `
        -MeanCiUpperPercent 1.0 -PairedP95DeltaPercent 0.0 -PairedBlocks 7 `
        -WarmupFrameCount 10000 -MeasureFrameCount 200000
    $shortFrames = Get-FormalGateVerdict -MeanCiLowerPercent -1.0 `
        -MeanCiUpperPercent 1.0 -PairedP95DeltaPercent 0.0 -PairedBlocks 8 `
        -WarmupFrameCount 9999 -MeasureFrameCount 199999
    if (
        $passingGate.Verdict -ne 'PASS' -or
        $p95Boundary.Verdict -ne 'FAIL' -or
        $shortRun.Verdict -ne 'INSUFFICIENT_SAMPLES' -or
        $shortFrames.Verdict -ne 'INSUFFICIENT_SAMPLES'
    ) {
        throw 'Internal formal-gate smoke test failed.'
    }
    if (
        (Convert-BackendName 'Dx12') -ne 'dx12' -or
        (Convert-BackendName 'BrowserWebGpu') -ne 'webgpu' -or
        (Convert-BackendName 'OpenGL') -ne 'gl'
    ) {
        throw 'Internal backend normalization smoke test failed.'
    }
}

function New-SourceSnapshot {
    param([Parameter(Mandatory)]$Revision)

    $snapshotPath = Join-Path $script:SnapshotRoot (
        '{0}-{1}' -f $Revision.Role, $Revision.ShortOid
    )
    $markerPath = Join-Path $snapshotPath '.perf-revision'
    Assert-SafeChildPath -Path $snapshotPath -Root $script:SnapshotRoot

    if (Test-Path -LiteralPath $snapshotPath) {
        $marker = if (Test-Path -LiteralPath $markerPath) {
            (Get-Content -LiteralPath $markerPath -Raw).Trim()
        } else {
            ''
        }
        if ($marker -eq $Revision.Oid) {
            Write-Host "Reusing immutable $($Revision.Role) snapshot: $snapshotPath"
            $Revision.SnapshotPath = $snapshotPath
            return
        }
        Remove-Item -LiteralPath $snapshotPath -Recurse -Force
    }

    New-Item -ItemType Directory -Path $snapshotPath -Force | Out-Null
    $archivePath = Join-Path $script:SnapshotRoot (
        '.{0}-{1}.tar' -f $Revision.Role, $Revision.ShortOid
    )
    Assert-SafeChildPath -Path $archivePath -Root $script:SnapshotRoot
    try {
        Write-Host "Exporting $($Revision.Ref) ($($Revision.ShortOid)) -> $snapshotPath"
        & $script:GitPath -C $script:RepoRoot archive --format=tar --output=$archivePath $Revision.Oid
        if ($LASTEXITCODE -ne 0) {
            throw "git archive failed for $($Revision.Ref)."
        }
        & $script:TarPath -xf $archivePath -C $snapshotPath
        if ($LASTEXITCODE -ne 0) {
            throw "tar extraction failed for $($Revision.Ref)."
        }
        Set-Content -LiteralPath $markerPath -Value $Revision.Oid -NoNewline -Encoding utf8
    } finally {
        if (Test-Path -LiteralPath $archivePath) {
            Remove-Item -LiteralPath $archivePath -Force
        }
    }
    $Revision.SnapshotPath = $snapshotPath
}

function Build-Revision {
    param([Parameter(Mandatory)]$Revision)

    $targetPath = Join-Path $script:BuildRoot (
        '{0}-{1}' -f $Revision.Role, $Revision.ShortOid
    )
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    $Revision.TargetPath = $targetPath

    $buildLog = Join-Path $script:ResultsRoot "build-$($Revision.Role).log"
    $featureCsv = $script:CanonicalFeatures -join ','
    Write-Host "Building $($Revision.Role) with --locked --no-default-features --features $featureCsv"

    $previousTarget = [Environment]::GetEnvironmentVariable('CARGO_TARGET_DIR', 'Process')
    $previousIncremental = [Environment]::GetEnvironmentVariable('CARGO_INCREMENTAL', 'Process')
    try {
        $env:CARGO_TARGET_DIR = $targetPath
        $env:CARGO_INCREMENTAL = '0'
        & $script:CargoPath build `
            --manifest-path (Join-Path $Revision.SnapshotPath 'Cargo.toml') `
            --locked `
            --release `
            --no-default-features `
            --features $featureCsv `
            --example helmet_gltf_fixed_probe 2>&1 |
            Tee-Object -FilePath $buildLog
        if ($LASTEXITCODE -ne 0) {
            throw "$($Revision.Role) build failed; see $buildLog"
        }
    } finally {
        [Environment]::SetEnvironmentVariable('CARGO_TARGET_DIR', $previousTarget, 'Process')
        [Environment]::SetEnvironmentVariable('CARGO_INCREMENTAL', $previousIncremental, 'Process')
    }

    $extension = if ($IsWindows) { '.exe' } else { '' }
    $binaryPath = Join-Path $targetPath "release/examples/helmet_gltf_fixed_probe$extension"
    if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) {
        throw "Expected benchmark binary was not produced: $binaryPath"
    }
    $Revision.BinaryPath = $binaryPath
}

function Set-ProcessEnvironmentValue {
    param(
        [Parameter(Mandatory)]$StartInfo,
        [Parameter(Mandatory)][string]$Name,
        [AllowEmptyString()][string]$Value = ''
    )

    if ([string]::IsNullOrEmpty($Value)) {
        $StartInfo.Environment.Remove($Name) | Out-Null
    } else {
        $StartInfo.Environment[$Name] = $Value
    }
}

function Invoke-ProbeProcess {
    param(
        [Parameter(Mandatory)]$Revision,
        [Parameter(Mandatory)]$Arm,
        [Parameter(Mandatory)][int]$RunIndex,
        [Parameter(Mandatory)][int]$Block,
        [Parameter(Mandatory)][string]$Pattern,
        [Parameter(Mandatory)][int]$Position
    )

    $stem = 'run-{0:D4}-block-{1:D3}-{2}-{3}' -f @(
        $RunIndex,
        $Block,
        $Arm.Name.ToLowerInvariant(),
        $Revision.Role
    )
    $stdoutPath = Join-Path $script:LogRoot "$stem.stdout.log"
    $stderrPath = Join-Path $script:LogRoot "$stem.stderr.log"

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Revision.BinaryPath
    $startInfo.WorkingDirectory = $Revision.SnapshotPath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    foreach ($name in $script:ProbeEnvironmentNames) {
        $startInfo.Environment.Remove($name) | Out-Null
    }
    Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_WARMUP_FRAMES' "$WarmupFrames"
    Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_MEASURE_FRAMES' "$MeasureFrames"
    Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_GPU_DRIVEN' $Arm.GpuDriven
    if ($Arm.DisableEnvironment) {
        Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_DISABLE_ENV' '1'
    }
    if ($Arm.Untextured) {
        Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_UNTEXTURED' '1'
    }
    if ($Arm.NoDraw) {
        Set-ProcessEnvironmentValue $startInfo 'MYTH_HELMET_PROBE_NO_DRAW' '1'
    }
    Set-ProcessEnvironmentValue $startInfo 'WGPU_BACKEND' $Backend
    Set-ProcessEnvironmentValue $startInfo 'RUST_LOG' 'myth_render::core::context=debug,wgpu_core=warn,wgpu_hal=warn'
    Set-ProcessEnvironmentValue $startInfo 'RUST_BACKTRACE' '1'

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $startedAt = [DateTimeOffset]::UtcNow
    try {
        if (-not $process.Start()) {
            throw "Failed to start $($Revision.Role) probe process."
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            try { $process.Kill($true) } catch { Write-Warning $_ }
            $process.WaitForExit()
            throw "Probe timed out after $ProcessTimeoutSeconds seconds; the benchmark window/process did not close."
        }
        [Threading.Tasks.Task]::WaitAll(
            [Threading.Tasks.Task[]]@($stdoutTask, $stderrTask)
        )
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        Set-Content -LiteralPath $stdoutPath -Value $stdout -NoNewline -Encoding utf8
        Set-Content -LiteralPath $stderrPath -Value $stderr -NoNewline -Encoding utf8

        if ($process.ExitCode -ne 0) {
            throw "Probe exited with code $($process.ExitCode); see $stdoutPath and $stderrPath"
        }
        if (-not $process.HasExited) {
            throw 'Probe returned without terminating its window process.'
        }

        $combined = $stdout + [Environment]::NewLine + $stderr
        if ($combined -match '(?m)^HELMET_FIXED_PROBE_ERROR\s') {
            throw "Probe reported HELMET_FIXED_PROBE_ERROR; see $stdoutPath and $stderrPath"
        }
        $resultLines = @(
            ($combined -split '\r?\n') |
                Where-Object { $_.StartsWith($script:ProbeResultPrefix, [StringComparison]::Ordinal) }
        )
        if ($resultLines.Count -ne 1) {
            throw "Expected exactly one HELMET_FIXED_PROBE result, found $($resultLines.Count); see $stdoutPath and $stderrPath"
        }
        $result = Convert-ProbeResultLine $resultLines[0]
        if ($result.Warmup -ne $WarmupFrames -or $result.Frames -ne $MeasureFrames) {
            throw "Probe reported unexpected frame counts in $($resultLines[0])"
        }

        $expectedEnvironment = -not $Arm.DisableEnvironment
        $expectedTextured = -not $Arm.Untextured
        $expectedDraw = -not $Arm.NoDraw
        if (
            $result.Environment -ne $expectedEnvironment -or
            $result.Textured -ne $expectedTextured -or
            $result.DrawModel -ne $expectedDraw -or
            $result.GpuDriven -ine $Arm.GpuDriven
        ) {
            throw "Probe configuration did not match arm '$($Arm.Name)': $($resultLines[0])"
        }

        $backendMatch = [regex]::Match($combined, '(?m)Backend:\s*(?<value>[^\r\n]+)')
        $deviceMatch = [regex]::Match($combined, '(?m)Device:\s*(?<value>[^\r\n]+)')
        if (-not $backendMatch.Success -or -not $deviceMatch.Success) {
            throw "Could not identify the WGPU adapter/backend from startup logs; see $stderrPath"
        }
        $observedBackend = $backendMatch.Groups['value'].Value.Trim()
        $observedAdapter = $deviceMatch.Groups['value'].Value.Trim()
        $normalizedObservedBackend = Convert-BackendName $observedBackend
        if ($normalizedObservedBackend -ne (Convert-BackendName $Backend)) {
            throw "Requested WGPU backend '$Backend', but the probe selected '$observedBackend'."
        }
        if (
            -not [string]::IsNullOrWhiteSpace($ExpectedAdapter) -and
            $observedAdapter.IndexOf($ExpectedAdapter, [StringComparison]::OrdinalIgnoreCase) -lt 0
        ) {
            throw "Observed adapter '$observedAdapter' does not contain expected text '$ExpectedAdapter'."
        }
        $adapterIdentity = "$observedBackend|$observedAdapter"
        if ($null -eq $script:ReferenceAdapterIdentity) {
            $script:ReferenceAdapterIdentity = $adapterIdentity
        } elseif (-not $adapterIdentity.Equals(
            $script:ReferenceAdapterIdentity,
            [StringComparison]::Ordinal
        )) {
            throw "Adapter drift detected: expected '$script:ReferenceAdapterIdentity', observed '$adapterIdentity'."
        }

        return [pscustomobject][ordered]@{
            RunIndex = $RunIndex
            Block = $Block
            Pattern = $Pattern
            Position = $Position
            Arm = $Arm.Name
            RevisionRole = $Revision.Role
            RevisionRef = $Revision.Ref
            RevisionOid = $Revision.Oid
            StartedUtc = $startedAt.ToString('O', $script:InvariantCulture)
            ExitCode = $process.ExitCode
            Backend = $observedBackend
            Adapter = $observedAdapter
            Warmup = $result.Warmup
            Frames = $result.Frames
            Environment = $result.Environment
            Textured = $result.Textured
            DrawModel = $result.DrawModel
            GpuDriven = $result.GpuDriven
            TotalMs = $result.TotalMs
            MeanUs = $result.MeanUs
            MedianUs = $result.MedianUs
            P95Us = $result.P95Us
            P99Us = $result.P99Us
            MaxUs = $result.MaxUs
            MeanFps = $result.MeanFps
            MedianFps = $result.MedianFps
            P95Fps = $result.P95Fps
            P99Fps = $result.P99Fps
            StdoutLog = $stdoutPath
            StderrLog = $stderrPath
        }
    } finally {
        $process.Dispose()
    }
}

function Write-RawResults {
    param([Parameter(Mandatory)]$Rows)

    @($Rows) | Export-Csv -LiteralPath $script:RawCsvPath -NoTypeInformation -Encoding utf8
}

function Write-Summaries {
    param(
        [Parameter(Mandatory)]$Rows,
        [Parameter(Mandatory)]$ArmConfigurations,
        [Parameter(Mandatory)][int]$BlockCount,
        [Parameter(Mandatory)][int]$WarmupFrameCount,
        [Parameter(Mandatory)][int]$MeasureFrameCount,
        [Parameter(Mandatory)][int]$BootstrapResampleCount,
        [Parameter(Mandatory)][uint32]$BootstrapBaseSeed
    )

    $aggregateRows = [Collections.Generic.List[object]]::new()
    foreach ($arm in $ArmConfigurations) {
        foreach ($role in @('baseline', 'candidate')) {
            $samples = @($Rows | Where-Object {
                $_.Arm -eq $arm.Name -and $_.RevisionRole -eq $role
            })
            $expectedCount = $BlockCount * 2
            if ($samples.Count -ne $expectedCount) {
                throw "Expected $expectedCount $role samples for $($arm.Name), found $($samples.Count)."
            }
            $totalFrames = ($samples | Measure-Object Frames -Sum).Sum
            $totalMs = ($samples | Measure-Object TotalMs -Sum).Sum
            $aggregateRows.Add([pscustomobject][ordered]@{
                Arm = $arm.Name
                RevisionRole = $role
                RevisionRef = $samples[0].RevisionRef
                RevisionOid = $samples[0].RevisionOid
                Samples = $samples.Count
                TotalFrames = $totalFrames
                AggregateMeanUs = $totalMs * 1000.0 / $totalFrames
                AggregateSampleP95Us = ($samples | Measure-Object P95Us -Average).Average
                AggregateSampleP99Us = ($samples | Measure-Object P99Us -Average).Average
            })
        }
    }
    $aggregatePath = Join-Path $script:ResultsRoot 'aggregate.csv'
    $aggregateRows | Export-Csv -LiteralPath $aggregatePath -NoTypeInformation -Encoding utf8

    $revisionCounts = @(
        $Rows |
            Group-Object RevisionRole |
            ForEach-Object {
                $first = $_.Group[0]
                [pscustomobject][ordered]@{
                    RevisionRole = $_.Name
                    RevisionRef = $first.RevisionRef
                    RevisionOid = $first.RevisionOid
                    Samples = $_.Count
                }
            } |
            Sort-Object RevisionRole
    )
    $revisionCountsPath = Join-Path $script:ResultsRoot 'revision-sample-counts.csv'
    $revisionCounts |
        Export-Csv -LiteralPath $revisionCountsPath -NoTypeInformation -Encoding utf8

    $pairedRows = [Collections.Generic.List[object]]::new()
    foreach ($arm in $ArmConfigurations) {
        for ($block = 1; $block -le $BlockCount; $block++) {
            $baseline = @($Rows | Where-Object {
                $_.Arm -eq $arm.Name -and $_.Block -eq $block -and
                $_.RevisionRole -eq 'baseline'
            })
            $candidate = @($Rows | Where-Object {
                $_.Arm -eq $arm.Name -and $_.Block -eq $block -and
                $_.RevisionRole -eq 'candidate'
            })
            if ($baseline.Count -ne 2 -or $candidate.Count -ne 2) {
                throw "Block $block/$($arm.Name) is not a complete ABBA/BAAB pair."
            }
            $baselineFrames = ($baseline | Measure-Object Frames -Sum).Sum
            $candidateFrames = ($candidate | Measure-Object Frames -Sum).Sum
            $baselineMeanUs = ($baseline | Measure-Object TotalMs -Sum).Sum * 1000.0 / $baselineFrames
            $candidateMeanUs = ($candidate | Measure-Object TotalMs -Sum).Sum * 1000.0 / $candidateFrames
            $logRatio = [Math]::Log($candidateMeanUs / $baselineMeanUs)
            $baselineP95Us = ($baseline | Measure-Object P95Us -Average).Average
            $candidateP95Us = ($candidate | Measure-Object P95Us -Average).Average
            $p95LogRatio = [Math]::Log($candidateP95Us / $baselineP95Us)
            $pairedRows.Add([pscustomobject][ordered]@{
                Arm = $arm.Name
                Block = $block
                Pattern = if (($block % 2) -eq 1) { 'ABBA' } else { 'BAAB' }
                BaselineMeanUs = $baselineMeanUs
                CandidateMeanUs = $candidateMeanUs
                LogRatio = $logRatio
                CandidateRatio = [Math]::Exp($logRatio)
                CandidateDeltaPercent = ([Math]::Exp($logRatio) - 1.0) * 100.0
                BaselineMeanProcessP95Us = $baselineP95Us
                CandidateMeanProcessP95Us = $candidateP95Us
                P95LogRatio = $p95LogRatio
                P95CandidateRatio = [Math]::Exp($p95LogRatio)
                P95CandidateDeltaPercent = ([Math]::Exp($p95LogRatio) - 1.0) * 100.0
            })
        }
    }
    $pairedPath = Join-Path $script:ResultsRoot 'paired-log-ratios.csv'
    $pairedRows | Export-Csv -LiteralPath $pairedPath -NoTypeInformation -Encoding utf8

    $pairedSummary = [Collections.Generic.List[object]]::new()
    $armOrdinal = 0
    foreach ($arm in $ArmConfigurations) {
        $pairs = @($pairedRows | Where-Object { $_.Arm -eq $arm.Name })
        $meanLog = ($pairs | Measure-Object LogRatio -Average).Average
        $sumSquared = 0.0
        foreach ($pair in $pairs) {
            $sumSquared += [Math]::Pow($pair.LogRatio - $meanLog, 2.0)
        }
        $logSd = if ($pairs.Count -gt 1) {
            [Math]::Sqrt($sumSquared / ($pairs.Count - 1))
        } else {
            [double]::NaN
        }
        $ratio = [Math]::Exp($meanLog)
        $meanP95Log = ($pairs | Measure-Object P95LogRatio -Average).Average
        $pairedP95Ratio = [Math]::Exp($meanP95Log)
        $pairedP95DeltaPercent = ($pairedP95Ratio - 1.0) * 100.0
        $logRatios = [double[]]@($pairs | ForEach-Object { [double]$_.LogRatio })
        $armSeedValue = (
            [uint64]$BootstrapBaseSeed + [uint64]$armOrdinal * [uint64]2654435761
        ) % [uint64]4294967296
        $armSeed = [uint32]$armSeedValue
        $bootstrap = Get-PairedLogBootstrapCi `
            -LogRatios $logRatios `
            -Resamples $BootstrapResampleCount `
            -Seed $armSeed
        $ciLowerPercent = ([Math]::Exp($bootstrap.LowerLogRatio) - 1.0) * 100.0
        $ciUpperPercent = ([Math]::Exp($bootstrap.UpperLogRatio) - 1.0) * 100.0

        $baselineAggregate = @($aggregateRows | Where-Object {
            $_.Arm -eq $arm.Name -and $_.RevisionRole -eq 'baseline'
        })[0]
        $candidateAggregate = @($aggregateRows | Where-Object {
            $_.Arm -eq $arm.Name -and $_.RevisionRole -eq 'candidate'
        })[0]
        $aggregateMeanDeltaPercent = (
            $candidateAggregate.AggregateMeanUs / $baselineAggregate.AggregateMeanUs - 1.0
        ) * 100.0
        # The probe publishes one p95 per process, not raw frame intervals. This
        # is therefore the candidate/baseline ratio of the mean process p95s;
        # the explicit name prevents it being mistaken for a pooled-frame p95.
        $aggregateSampleP95DeltaPercent = (
            $candidateAggregate.AggregateSampleP95Us /
                $baselineAggregate.AggregateSampleP95Us - 1.0
        ) * 100.0
        $gate = Get-FormalGateVerdict `
            -MeanCiLowerPercent $ciLowerPercent `
            -MeanCiUpperPercent $ciUpperPercent `
            -PairedP95DeltaPercent $pairedP95DeltaPercent `
            -PairedBlocks $pairs.Count `
            -WarmupFrameCount $WarmupFrameCount `
            -MeasureFrameCount $MeasureFrameCount
        $pairedSummary.Add([pscustomobject][ordered]@{
            Arm = $arm.Name
            PairedBlocks = $pairs.Count
            MeanLogRatio = $meanLog
            LogRatioSampleSd = $logSd
            CandidateGeometricRatio = $ratio
            CandidateDeltaPercent = ($ratio - 1.0) * 100.0
            MeanP95LogRatio = $meanP95Log
            PairedP95GeometricRatio = $pairedP95Ratio
            PairedP95DeltaPercent = $pairedP95DeltaPercent
            BootstrapResamples = $BootstrapResampleCount
            BootstrapSeed = $armSeed
            Bootstrap95LowerLogRatio = $bootstrap.LowerLogRatio
            Bootstrap95UpperLogRatio = $bootstrap.UpperLogRatio
            Bootstrap95LowerDeltaPercent = $ciLowerPercent
            Bootstrap95UpperDeltaPercent = $ciUpperPercent
            BaselineAggregateMeanUs = $baselineAggregate.AggregateMeanUs
            CandidateAggregateMeanUs = $candidateAggregate.AggregateMeanUs
            AggregateMeanDeltaPercent = $aggregateMeanDeltaPercent
            BaselineAggregateSampleP95Us = $baselineAggregate.AggregateSampleP95Us
            CandidateAggregateSampleP95Us = $candidateAggregate.AggregateSampleP95Us
            AggregateSampleP95DeltaPercent = $aggregateSampleP95DeltaPercent
            MeanCiWithinPlusMinus2Percent = $gate.MeanPassed
            PairedP95RegressionBelow5Percent = $gate.P95Passed
            FormalGateEligible = $gate.Eligible
            GateVerdict = $gate.Verdict
        })
        $armOrdinal++
    }
    $pairedSummaryPath = Join-Path $script:ResultsRoot 'paired-summary.csv'
    $pairedSummary | Export-Csv -LiteralPath $pairedSummaryPath -NoTypeInformation -Encoding utf8

    Write-Host ''
    Write-Host 'Raw process samples per revision:'
    $revisionCounts |
        Select-Object RevisionRole, Samples, RevisionRef, RevisionOid |
        Format-Table -AutoSize |
        Out-String |
        Write-Host

    Write-Host 'Aggregate endpoint (sum(total_ms) / sum(frames)):'
    $aggregateRows |
        Select-Object Arm, RevisionRole, Samples, TotalFrames,
            @{ Name = 'AggregateMeanUs'; Expression = { '{0:F3}' -f $_.AggregateMeanUs } } |
        Format-Table -AutoSize |
        Out-String |
        Write-Host

    Write-Host 'Paired block log-ratio summary (candidate / baseline):'
    $pairedSummary |
        Select-Object Arm, PairedBlocks,
            @{ Name = 'MeanDelta%'; Expression = { '{0:+0.000;-0.000;0.000}' -f $_.CandidateDeltaPercent } },
            @{ Name = 'Bootstrap95%'; Expression = {
                '[{0:+0.000;-0.000;0.000}, {1:+0.000;-0.000;0.000}]' -f
                    $_.Bootstrap95LowerDeltaPercent, $_.Bootstrap95UpperDeltaPercent
            } },
            @{ Name = 'PairedP95Delta%'; Expression = {
                '{0:+0.000;-0.000;0.000}' -f $_.PairedP95DeltaPercent
            } },
            GateVerdict |
        Format-Table -AutoSize |
        Out-String |
        Write-Host
    Write-Host 'Gate: paired mean-frame-time bootstrap 95% CI must stay in [-2%, +2%], and paired block-p95 regression must be <5%.'
    Write-Host 'Formal PASS/FAIL requires at least eight paired blocks, 10,000 warmup frames, and 200,000 measured frames per process.'
    Write-Host "Raw samples: $script:RawCsvPath"
    Write-Host "Run config:  $script:RunConfigurationPath"
    Write-Host "Counts:      $revisionCountsPath"
    Write-Host "Aggregate:  $aggregatePath"
    Write-Host "Pairs:      $pairedPath"
    Write-Host "Summary:    $pairedSummaryPath"
}

function Convert-CsvBoolean {
    param(
        [Parameter(Mandatory)][AllowEmptyString()][string]$Value,
        [Parameter(Mandatory)][string]$FieldName
    )

    $parsed = $false
    if (-not [bool]::TryParse($Value, [ref]$parsed)) {
        throw "Raw result field '$FieldName' is not a boolean: '$Value'."
    }
    return $parsed
}

function Assert-RecordedRunIntegrity {
    param(
        [Parameter(Mandatory)]$Rows,
        [Parameter(Mandatory)]$Configuration
    )

    $blockCount = [int]$Configuration.Blocks
    $warmupFrameCount = [int]$Configuration.WarmupFrames
    $measureFrameCount = [int]$Configuration.MeasureFrames
    $arms = @($Configuration.Arms)
    if ($blockCount -lt 1 -or $warmupFrameCount -lt 0 -or $measureFrameCount -lt 1) {
        throw 'Recorded run configuration contains invalid frame or block counts.'
    }
    if ($arms.Count -eq 0) {
        throw 'Recorded run configuration contains no workload arms.'
    }
    if (@($arms.Name | Sort-Object -Unique).Count -ne $arms.Count) {
        throw 'Recorded run configuration contains duplicate workload arms.'
    }

    $expectedRowCount = $blockCount * $arms.Count * 4
    if (@($Rows).Count -ne $expectedRowCount) {
        throw "Recorded run is incomplete: expected $expectedRowCount raw rows, found $(@($Rows).Count)."
    }

    $seenCells = [Collections.Generic.HashSet[string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    $seenRunIndices = [Collections.Generic.HashSet[int]]::new()
    $adapterIdentity = $null
    foreach ($row in @($Rows)) {
        $runIndex = [int]::Parse([string]$row.RunIndex, $script:InvariantCulture)
        $block = [int]::Parse([string]$row.Block, $script:InvariantCulture)
        $position = [int]::Parse([string]$row.Position, $script:InvariantCulture)
        $warmup = [int]::Parse([string]$row.Warmup, $script:InvariantCulture)
        $frames = [int]::Parse([string]$row.Frames, $script:InvariantCulture)
        $exitCode = [int]::Parse([string]$row.ExitCode, $script:InvariantCulture)
        if ($runIndex -lt 1 -or $runIndex -gt $expectedRowCount -or -not $seenRunIndices.Add($runIndex)) {
            throw "Raw result contains an invalid or duplicate RunIndex '$runIndex'."
        }
        if ($block -lt 1 -or $block -gt $blockCount -or $position -lt 1 -or $position -gt 4) {
            throw "Raw result contains an invalid block/position pair: $block/$position."
        }
        if ($exitCode -ne 0) {
            throw "Raw result RunIndex $runIndex records non-zero exit code $exitCode."
        }
        if ($warmup -ne $warmupFrameCount -or $frames -ne $measureFrameCount) {
            throw "Raw result RunIndex $runIndex does not match recorded frame counts."
        }

        $expectedPattern = if (($block % 2) -eq 1) { 'ABBA' } else { 'BAAB' }
        $expectedRole = @(Get-BlockPattern $block)[$position - 1]
        if ([string]$row.Pattern -cne $expectedPattern -or [string]$row.RevisionRole -cne $expectedRole) {
            throw "Raw result RunIndex $runIndex violates the recorded ABBA/BAAB schedule."
        }

        $armMatches = @($arms | Where-Object { $_.Name -ceq [string]$row.Arm })
        if ($armMatches.Count -ne 1) {
            throw "Raw result RunIndex $runIndex names unknown arm '$($row.Arm)'."
        }
        $arm = $armMatches[0]
        $cell = "$block|$($arm.Name)|$position"
        if (-not $seenCells.Add($cell)) {
            throw "Raw result duplicates schedule cell '$cell'."
        }

        $revision = if ($expectedRole -eq 'baseline') {
            $Configuration.Baseline
        } else {
            $Configuration.Candidate
        }
        if (
            [string]$row.RevisionRef -cne [string]$revision.Ref -or
            [string]$row.RevisionOid -cne [string]$revision.Oid
        ) {
            throw "Raw result RunIndex $runIndex does not match the recorded $expectedRole revision."
        }

        $expectedEnvironment = -not [bool]$arm.DisableEnvironment
        $expectedTextured = -not [bool]$arm.Untextured
        $expectedDraw = -not [bool]$arm.NoDraw
        if (
            (Convert-CsvBoolean ([string]$row.Environment) 'Environment') -ne $expectedEnvironment -or
            (Convert-CsvBoolean ([string]$row.Textured) 'Textured') -ne $expectedTextured -or
            (Convert-CsvBoolean ([string]$row.DrawModel) 'DrawModel') -ne $expectedDraw -or
            [string]$row.GpuDriven -ine [string]$arm.GpuDriven
        ) {
            throw "Raw result RunIndex $runIndex does not match recorded arm '$($arm.Name)'."
        }

        if ((Convert-BackendName ([string]$row.Backend)) -ne
            (Convert-BackendName ([string]$Configuration.RequestedBackend))
        ) {
            throw "Raw result RunIndex $runIndex used unexpected backend '$($row.Backend)'."
        }
        $adapter = [string]$row.Adapter
        if ([string]::IsNullOrWhiteSpace($adapter)) {
            throw "Raw result RunIndex $runIndex has no adapter identity."
        }
        if ($null -eq $adapterIdentity) {
            $adapterIdentity = $adapter
        } elseif (-not $adapter.Equals($adapterIdentity, [StringComparison]::Ordinal)) {
            throw "Raw results mix adapters '$adapterIdentity' and '$adapter'."
        }
        $expectedAdapter = [string]$Configuration.ExpectedAdapterSubstring
        if (
            -not [string]::IsNullOrWhiteSpace($expectedAdapter) -and
            $adapter.IndexOf($expectedAdapter, [StringComparison]::OrdinalIgnoreCase) -lt 0
        ) {
            throw "Raw result adapter '$adapter' does not contain recorded expectation '$expectedAdapter'."
        }

        foreach ($fieldName in @(
            'TotalMs', 'MeanUs', 'MedianUs', 'P95Us', 'P99Us', 'MaxUs',
            'MeanFps', 'MedianFps', 'P95Fps', 'P99Fps'
        )) {
            $value = [double]::Parse(
                [string]$row.$fieldName,
                $script:InvariantCulture
            )
            if (-not [double]::IsFinite($value) -or $value -le 0.0) {
                throw "Raw result RunIndex $runIndex contains invalid $fieldName value."
            }
        }
    }
}

function Get-RecordedEnvironmentComparison {
    param(
        [Parameter(Mandatory)]$Configuration,
        [Parameter(Mandatory)][string]$ResultsDirectory
    )

    $endPath = Join-Path $ResultsDirectory 'run-environment-end.json'
    if (-not (Test-Path -LiteralPath $endPath -PathType Leaf)) {
        throw "Recorded run is incomplete: missing $endPath"
    }
    $end = Get-Content -LiteralPath $endPath -Raw | ConvertFrom-Json
    $startFingerprint = $Configuration.HostEnvironmentAtStart |
        ConvertTo-Json -Depth 6 -Compress
    $endFingerprint = $end | ConvertTo-Json -Depth 6 -Compress
    return [pscustomobject]@{
        Stable = $startFingerprint -ceq $endFingerprint
        EndPath = $endPath
    }
}

function Get-EligibleGateFailures {
    param([Parameter(Mandatory)][string]$SummaryPath)

    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Performance summary was not produced: $SummaryPath"
    }
    return @(
        Import-Csv -LiteralPath $SummaryPath | Where-Object {
            $_.FormalGateEligible -eq 'True' -and $_.GateVerdict -eq 'FAIL'
        }
    )
}

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if ($AnalyzeOnly -and $PlanOnly) {
    throw 'AnalyzeOnly and PlanOnly are mutually exclusive.'
}
if ($AnalyzeOnly) {
    if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
        throw 'AnalyzeOnly requires -OutputDirectory containing a completed run.'
    }
    Assert-ProbeParser
    Assert-StatisticsHelpers

    $ResultsRoot = Resolve-FullPath -Path $OutputDirectory -BasePath $RepoRoot
    $RunConfigurationPath = Join-Path $ResultsRoot 'run-configuration.json'
    $RawCsvPath = Join-Path $ResultsRoot 'raw.csv'
    if (-not (Test-Path -LiteralPath $RunConfigurationPath -PathType Leaf)) {
        throw "AnalyzeOnly could not find $RunConfigurationPath"
    }
    if (-not (Test-Path -LiteralPath $RawCsvPath -PathType Leaf)) {
        throw "AnalyzeOnly could not find $RawCsvPath"
    }

    $configuration = Get-Content -LiteralPath $RunConfigurationPath -Raw |
        ConvertFrom-Json
    $rows = @(Import-Csv -LiteralPath $RawCsvPath)
    Assert-RecordedRunIntegrity -Rows $rows -Configuration $configuration
    $environment = Get-RecordedEnvironmentComparison `
        -Configuration $configuration `
        -ResultsDirectory $ResultsRoot
    if (-not $environment.Stable) {
        Write-Host 'Recorded environment drifted during the benchmark; results are invalid.' -ForegroundColor Red
        Write-Host "End environment: $($environment.EndPath)"
        exit 4
    }

    $recordedArms = @($configuration.Arms)
    $recordedBlocks = [int]$configuration.Blocks
    $recordedWarmupFrames = [int]$configuration.WarmupFrames
    $recordedMeasureFrames = [int]$configuration.MeasureFrames
    $recordedBootstrapResamples = [int]$configuration.Bootstrap.Resamples
    $recordedBootstrapSeed = [uint32]$configuration.Bootstrap.BaseSeed
    Write-Host "AnalyzeOnly baseline:  $($configuration.Baseline.Ref) -> $($configuration.Baseline.Oid)"
    Write-Host "AnalyzeOnly candidate: $($configuration.Candidate.Ref) -> $($configuration.Candidate.Oid)"
    Write-Host (
        'Recorded workload: warmup={0} measure={1} blocks={2} arms={3}' -f
            $recordedWarmupFrames,
            $recordedMeasureFrames,
            $recordedBlocks,
            ($recordedArms.Name -join ',')
    )
    Write-Summaries `
        -Rows $rows `
        -ArmConfigurations $recordedArms `
        -BlockCount $recordedBlocks `
        -WarmupFrameCount $recordedWarmupFrames `
        -MeasureFrameCount $recordedMeasureFrames `
        -BootstrapResampleCount $recordedBootstrapResamples `
        -BootstrapBaseSeed $recordedBootstrapSeed
    Write-Host 'AnalyzeOnly: the completed run was integrity-checked and reprocessed; no build or GPU process was started.'

    $pairedSummaryPath = Join-Path $ResultsRoot 'paired-summary.csv'
    $eligibleFailures = @(Get-EligibleGateFailures -SummaryPath $pairedSummaryPath)
    if ($eligibleFailures.Count -ne 0) {
        Write-Host 'One or more eligible workload arms failed the formal performance gate.' -ForegroundColor Red
        $eligibleFailures | Select-Object Arm, GateVerdict |
            Format-Table -AutoSize | Out-String | Write-Host
        exit 3
    }
    exit 0
}

$GitPath = Get-NativeCommandPath 'git'
$TarPath = Get-NativeCommandPath 'tar'
$CargoPath = Get-NativeCommandPath 'cargo'
$SnapshotRoot = Join-Path $RepoRoot 'target/perf-snapshots'
$BuildRoot = Join-Path $RepoRoot 'target/perf-builds'
$ReferenceAdapterIdentity = $null

$CanonicalFeatures = @(
    $Features -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
)
if ($CanonicalFeatures.Count -eq 0) {
    throw 'At least one explicit Cargo feature is required.'
}
if (@($CanonicalFeatures | Where-Object { $_ -ieq 'rdg_bench_support' }).Count -ne 0) {
    throw 'rdg_bench_support must not be enabled for the timing comparison.'
}
$CanonicalFeatures = @($CanonicalFeatures | Sort-Object -Unique)
$ArmConfigurations = @(
    $Arms -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ } |
        ForEach-Object { Get-ArmConfiguration $_ }
)
if ($ArmConfigurations.Count -eq 0) {
    throw 'At least one workload arm is required.'
}
if (@($ArmConfigurations.Name | Sort-Object -Unique).Count -ne $ArmConfigurations.Count) {
    throw 'Each workload arm may be specified only once.'
}

Assert-ProbeParser
Assert-StatisticsHelpers
$baseline = Resolve-Revision -Role baseline -Ref $BaselineRef
$candidate = Resolve-Revision -Role candidate -Ref $CandidateRef
if (
    -not $PlanOnly -and
    $CandidateRef.Equals('HEAD', [StringComparison]::OrdinalIgnoreCase)
) {
    $dirtyPaths = @(Invoke-GitText @('status', '--porcelain'))
    $blockingDirtyPaths = @(
        $dirtyPaths | Where-Object {
            $AllowedLocalOnlyStatusEntries -cnotcontains [string]$_
        }
    )
    if ($blockingDirtyPaths.Count -ne 0) {
        throw (
            'CandidateRef HEAD names only the committed tree, but the worktree contains candidate-affecting changes. ' +
            'Commit the reviewed candidate and pass its exact OID. Blocking entries: ' +
            ($blockingDirtyPaths -join ', ')
        )
    }
}
$revisions = @{
    baseline = $baseline
    candidate = $candidate
}

Write-Host "Baseline:  $($baseline.Ref) -> $($baseline.Oid)"
Write-Host "Candidate: $($candidate.Ref) -> $($candidate.Oid)"
Write-Host "Features:  --no-default-features --features $($CanonicalFeatures -join ',')"
Write-Host "Backend:   $Backend"
Write-Host "Workload:  warmup=$WarmupFrames measure=$MeasureFrames blocks=$Blocks arms=$($ArmConfigurations.Name -join ',')"
Write-Host "Schedule:  $($Blocks * $ArmConfigurations.Count * 4) serial processes; odd blocks ABBA, even blocks BAAB"

if ($PlanOnly) {
    Write-Host 'PlanOnly: refs, probe availability, parameters, schedule, result parser, and deterministic bootstrap helpers validated; no snapshots, builds, or GPU processes were started.'
    exit 0
}

$timestamp = [DateTimeOffset]::Now.ToString('yyyyMMdd-HHmmss', $InvariantCulture)
$ResultsRoot = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $RepoRoot "target/perf-results/render-world-$timestamp"
} else {
    Resolve-FullPath -Path $OutputDirectory -BasePath $RepoRoot
}
$RunConfigurationPath = Join-Path $ResultsRoot 'run-configuration.json'
if (Test-Path -LiteralPath (Join-Path $ResultsRoot 'raw.csv')) {
    throw "Output directory already contains raw.csv: $ResultsRoot"
}
$LogRoot = Join-Path $ResultsRoot 'logs'
$RawCsvPath = Join-Path $ResultsRoot 'raw.csv'
New-Item -ItemType Directory -Path $SnapshotRoot, $BuildRoot, $LogRoot -Force | Out-Null
$hostEnvironmentStart = Get-HostEnvironmentSnapshot
[ordered]@{
    CreatedUtc = [DateTimeOffset]::UtcNow.ToString('O', $InvariantCulture)
    Baseline = [ordered]@{ Ref = $baseline.Ref; Oid = $baseline.Oid }
    Candidate = [ordered]@{ Ref = $candidate.Ref; Oid = $candidate.Oid }
    WarmupFrames = $WarmupFrames
    MeasureFrames = $MeasureFrames
    Blocks = $Blocks
    Arms = @($ArmConfigurations | ForEach-Object {
        [ordered]@{
            Name = $_.Name
            DisableEnvironment = $_.DisableEnvironment
            Untextured = $_.Untextured
            NoDraw = $_.NoDraw
            GpuDriven = $_.GpuDriven
        }
    })
    Cargo = [ordered]@{
        Locked = $true
        NoDefaultFeatures = $true
        Features = $CanonicalFeatures
        Profile = 'release'
    }
    RequestedBackend = $Backend
    ExpectedAdapterSubstring = $ExpectedAdapter
    LogicalWindowSize = '1280x720 (fixed probe/App default)'
    HostEnvironmentAtStart = $hostEnvironmentStart
    ProcessTimeoutSeconds = $ProcessTimeoutSeconds
    Bootstrap = [ordered]@{
        Unit = 'paired ABBA/BAAB block log-ratio'
        Method = 'percentile'
        Resamples = $BootstrapResamples
        BaseSeed = $BootstrapSeed
        Generator = 'LCG(1664525,1013904223,2^32)'
        IndexMapping = 'multiply-high(state, block_count)'
    }
    FormalGate = [ordered]@{
        MinimumPairedBlocks = 8
        MinimumWarmupFramesPerProcess = 10000
        MinimumMeasuredFramesPerProcess = 200000
        MeanBootstrap95CiLowerPercent = -2.0
        MeanBootstrap95CiUpperPercent = 2.0
        PairedBlockP95RegressionStrictlyBelowPercent = 5.0
    }
} | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $runConfigurationPath -Encoding utf8

New-SourceSnapshot $baseline
New-SourceSnapshot $candidate
Build-Revision $baseline
Build-Revision $candidate

$rows = [Collections.Generic.List[object]]::new()
$runIndex = 0
for ($block = 1; $block -le $Blocks; $block++) {
    $order = @(Get-BlockPattern $block)
    $pattern = if (($block % 2) -eq 1) { 'ABBA' } else { 'BAAB' }
    foreach ($arm in $ArmConfigurations) {
        for ($position = 0; $position -lt $order.Count; $position++) {
            $runIndex++
            $role = $order[$position]
            $revision = $revisions[$role]
            Write-Host (
                '[{0}/{1}] block={2} pattern={3} position={4} arm={5} revision={6} ({7})' -f
                $runIndex,
                ($Blocks * $ArmConfigurations.Count * 4),
                $block,
                $pattern,
                ($position + 1),
                $arm.Name,
                $role,
                $revision.ShortOid
            )
            $row = Invoke-ProbeProcess `
                -Revision $revision `
                -Arm $arm `
                -RunIndex $runIndex `
                -Block $block `
                -Pattern $pattern `
                -Position ($position + 1)
            $rows.Add($row)
            Write-RawResults $rows
            Write-Host (
                '  mean={0:F3} us p95={1:F3} us adapter={2} ({3})' -f
                $row.MeanUs, $row.P95Us, $row.Adapter, $row.Backend
            )
        }
    }
}

$recordedConfiguration = Get-Content -LiteralPath $RunConfigurationPath -Raw |
    ConvertFrom-Json
Assert-RecordedRunIntegrity -Rows $rows -Configuration $recordedConfiguration
Write-Summaries `
    -Rows $rows `
    -ArmConfigurations $ArmConfigurations `
    -BlockCount $Blocks `
    -WarmupFrameCount $WarmupFrames `
    -MeasureFrameCount $MeasureFrames `
    -BootstrapResampleCount $BootstrapResamples `
    -BootstrapBaseSeed $BootstrapSeed

$hostEnvironmentEnd = Get-HostEnvironmentSnapshot
$hostEnvironmentEndPath = Join-Path $ResultsRoot 'run-environment-end.json'
$hostEnvironmentEnd | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath $hostEnvironmentEndPath -Encoding utf8
$startFingerprint = $hostEnvironmentStart | ConvertTo-Json -Depth 6 -Compress
$endFingerprint = $hostEnvironmentEnd | ConvertTo-Json -Depth 6 -Compress
if ($startFingerprint -cne $endFingerprint) {
    Write-Host 'Formal environment drifted during the benchmark; results are invalid.' -ForegroundColor Red
    Write-Host "End environment: $hostEnvironmentEndPath"
    exit 4
}

$pairedSummaryPath = Join-Path $ResultsRoot 'paired-summary.csv'
$eligibleFailures = @(Get-EligibleGateFailures -SummaryPath $pairedSummaryPath)
if ($eligibleFailures.Count -ne 0) {
    Write-Host 'One or more eligible workload arms failed the formal performance gate.' -ForegroundColor Red
    $eligibleFailures | Select-Object Arm, GateVerdict |
        Format-Table -AutoSize | Out-String | Write-Host
    exit 3
}
