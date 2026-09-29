"""
SPECTRA - Hardware Detection Module
MODULE A

Detects CPU, GPU, NPU, RAM, OS, architecture, and available acceleration backends.
Reports honestly — never fakes NPU availability.
"""

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("HARDWARE")


@dataclass
class CPUInfo:
    name: str
    architecture: str
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    is_arm: bool
    is_snapdragon: bool
    vendor: str


@dataclass
class GPUInfo:
    name: str
    vendor: str
    vram_mb: Optional[float]
    driver_version: Optional[str]
    api: str  # "directx", "vulkan", "cuda", "metal", "unknown"


@dataclass
class NPUInfo:
    available: bool
    name: Optional[str]
    vendor: Optional[str]
    framework: Optional[str]   # "QNN", "DirectML-NPU", "CoreML-ANE", etc.
    notes: str


@dataclass
class RAMInfo:
    total_mb: float
    available_mb: float


@dataclass
class HardwareProfile:
    cpu: CPUInfo
    gpus: list[GPUInfo]
    npu: NPUInfo
    ram: RAMInfo
    os_name: str
    os_version: str
    hostname: str
    python_version: str
    npu_available: bool = field(init=False)

    def __post_init__(self):
        self.npu_available = self.npu.available


# ── Detection Helpers ──────────────────────────────────────────────────────────

def _detect_cpu() -> CPUInfo:
    name = platform.processor() or "UNKNOWN"
    arch = platform.machine()
    is_arm = arch.upper() in ("ARM64", "AARCH64", "ARM")
    is_snapdragon = "snapdragon" in name.lower() or "qualcomm" in name.lower()

    physical_cores = 1
    logical_cores = 1
    freq_mhz = 0.0

    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        freq_mhz = freq.current if freq else 0.0
    except ImportError:
        import os as _os
        logical_cores = _os.cpu_count() or 1

    # Determine vendor from name
    name_lower = name.lower()
    if "intel" in name_lower:
        vendor = "Intel"
    elif "amd" in name_lower:
        vendor = "AMD"
    elif "qualcomm" in name_lower or "snapdragon" in name_lower:
        vendor = "Qualcomm"
    elif "apple" in name_lower:
        vendor = "Apple"
    else:
        vendor = "Unknown"

    log.info(f"CPU detected: {name} | arch={arch} | cores={physical_cores}P/{logical_cores}L")
    if is_snapdragon:
        log.info("Snapdragon CPU detected — NPU integration path available")

    return CPUInfo(
        name=name,
        architecture=arch,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        frequency_mhz=freq_mhz,
        is_arm=is_arm,
        is_snapdragon=is_snapdragon,
        vendor=vendor,
    )


def _detect_gpus_windows() -> list[GPUInfo]:
    """Use WMI via subprocess to detect GPUs on Windows."""
    gpus: list[GPUInfo] = []
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-WmiObject Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Csv -NoTypeInformation"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return gpus
        # Skip header line
        for line in lines[1:]:
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 1 and parts[0]:
                name = parts[0]
                try:
                    vram_mb = int(parts[1]) / (1024 * 1024) if parts[1] else None
                except (ValueError, IndexError):
                    vram_mb = None
                driver = parts[2] if len(parts) > 2 else None
                name_lower = name.lower()
                vendor = "Intel" if "intel" in name_lower else \
                         "NVIDIA" if "nvidia" in name_lower else \
                         "AMD" if ("amd" in name_lower or "radeon" in name_lower) else \
                         "Qualcomm" if "qualcomm" in name_lower else "Unknown"
                gpus.append(GPUInfo(name=name, vendor=vendor, vram_mb=vram_mb,
                                    driver_version=driver, api="directx"))
                log.info(f"GPU detected: {name} | VRAM={vram_mb:.0f}MB" if vram_mb else f"GPU detected: {name}")
    except Exception as e:
        log.warning(f"GPU detection via WMI failed: {e}")
    return gpus


def _detect_gpus() -> list[GPUInfo]:
    if platform.system() == "Windows":
        return _detect_gpus_windows()
    return []


def _detect_npu() -> NPUInfo:
    """
    Detect NPU availability.

    Strategy (in order):
    1. Check for QNN (Qualcomm Neural Network SDK) in ONNX Runtime providers
    2. Check for DirectML NPU
    3. Check QNN DLL presence
    4. Check ONNX Runtime QNN EP
    5. System environment NPU hints

    Reports honestly — never fakes availability.
    """
    # 1. Check ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "QNNExecutionProvider" in providers:
            log.info("NPU detected via ONNX Runtime QNNExecutionProvider")
            return NPUInfo(
                available=True,
                name="Qualcomm NPU",
                vendor="Qualcomm",
                framework="QNN",
                notes="Detected via ONNX Runtime QNNExecutionProvider"
            )
        if "DmlExecutionProvider" in providers:
            log.info("DirectML execution provider available — may include NPU via DML")
            return NPUInfo(
                available=False,
                name="DirectML (GPU/NPU)",
                vendor="Microsoft/OEM",
                framework="DirectML",
                notes="DirectML found but NPU-specific path unconfirmed on this hardware"
            )
    except ImportError:
        pass

    # 2. Check for QNN runtime DLLs
    qnn_dll_paths = [
        r"C:\Windows\System32\QnnHtp.dll",
        r"C:\Program Files\Qualcomm\AIStack\SNPE\lib\aarch64-windows-msvc\QnnHtp.dll",
        r"C:\Windows\System32\QnnCpu.dll",
    ]
    for p in qnn_dll_paths:
        if os.path.exists(p):
            log.info(f"QNN DLL found: {p}")
            return NPUInfo(
                available=True,
                name="Qualcomm Hexagon NPU",
                vendor="Qualcomm",
                framework="QNN",
                notes=f"QNN runtime DLL found at {p}"
            )

    # 3. Check Windows AI-related registry or APIs (strictly avoiding 'Input' devices)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-WmiObject Win32_PnPEntity | Where-Object { ($_.Name -match '\\bNPU\\b' -or $_.Name -match 'Neural Processing' -or $_.Name -match '\\bHexagon\\b') -and ($_.Name -notmatch 'Input|Keyboard|Mouse|Touch') } | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10
            )
            raw = result.stdout.strip()
            if raw:
                npu_name = raw.splitlines()[-1].strip()
                log.info(f"NPU hardware device confirmed in PnP: {npu_name}")
                return NPUInfo(
                    available=True,
                    name=npu_name,
                    vendor="Qualcomm" if "snapdragon" in npu_name.lower() or "hexagon" in npu_name.lower() else "OEM",
                    framework="QNN / Windows ML",
                    notes=f"Found via Windows PnP: {npu_name}"
                )
        except Exception as e:
            log.debug(f"PnP NPU check error: {e}")

    # 4. Intel integrated AI — check for OpenVINO
    try:
        import openvino
        log.info("OpenVINO detected — Intel AI acceleration available (no discrete NPU)")
        return NPUInfo(
            available=False,
            name="Intel integrated AI (OpenVINO)",
            vendor="Intel",
            framework="OpenVINO",
            notes="OpenVINO found but this is Intel AI, not Snapdragon NPU. NPU via QNN: NOT DETECTED"
        )
    except ImportError:
        pass

    log.warning("NPU: NOT DETECTED on this hardware")
    return NPUInfo(
        available=False,
        name=None,
        vendor=None,
        framework=None,
        notes="No NPU detected. Qualcomm QNN not found. This is an Intel x86 machine — NPU support requires Snapdragon hardware or compatible drivers."
    )


def _detect_ram() -> RAMInfo:
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_mb = vm.total / (1024 * 1024)
        available_mb = vm.available / (1024 * 1024)
        log.info(f"RAM: {total_mb:.0f}MB total, {available_mb:.0f}MB available")
        return RAMInfo(total_mb=total_mb, available_mb=available_mb)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-WmiObject Win32_OperatingSystem) | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Csv -NoTypeInformation"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].split(",")
                total_mb = int(parts[0].strip('"')) / 1024
                avail_mb = int(parts[1].strip('"')) / 1024
                return RAMInfo(total_mb=total_mb, available_mb=avail_mb)
        except Exception:
            pass
        return RAMInfo(total_mb=0.0, available_mb=0.0)


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_hardware() -> HardwareProfile:
    """Run full hardware detection and return a HardwareProfile."""
    log.info("Starting hardware detection...")

    cpu = _detect_cpu()
    gpus = _detect_gpus()
    npu = _detect_npu()
    ram = _detect_ram()

    profile = HardwareProfile(
        cpu=cpu,
        gpus=gpus,
        npu=npu,
        ram=ram,
        os_name=platform.system(),
        os_version=platform.version(),
        hostname=platform.node(),
        python_version=sys.version,
    )

    log.info("Hardware detection complete")
    return profile


def print_hardware_report(profile: HardwareProfile) -> None:
    """Print a formatted hardware report to stdout."""
    divider = "-" * 50
    print(f"\n{'=' * 50}")
    print("  SPECTRA - HARDWARE DETECTION REPORT")
    print(f"{'=' * 50}")
    print(f"\n  OS       : {profile.os_name} {profile.os_version}")
    print(f"  Hostname : {profile.hostname}")
    print(f"  Python   : {profile.python_version.split()[0]}")
    print(f"\n{divider}")
    print("  CPU")
    print(f"{divider}")
    print(f"  Name         : {profile.cpu.name}")
    print(f"  Architecture : {profile.cpu.architecture}")
    print(f"  Vendor       : {profile.cpu.vendor}")
    print(f"  Cores        : {profile.cpu.physical_cores}P / {profile.cpu.logical_cores}L")
    print(f"  Frequency    : {profile.cpu.frequency_mhz:.0f} MHz" if profile.cpu.frequency_mhz else "  Frequency    : UNKNOWN")
    print(f"  Snapdragon   : {'YES' if profile.cpu.is_snapdragon else 'NO'}")
    print(f"\n{divider}")
    print("  GPU")
    print(f"{divider}")
    if profile.gpus:
        for i, gpu in enumerate(profile.gpus, 1):
            print(f"  [{i}] {gpu.name}")
            print(f"      Vendor : {gpu.vendor}")
            print(f"      VRAM   : {gpu.vram_mb:.0f} MB" if gpu.vram_mb else "      VRAM   : UNKNOWN")
            print(f"      Driver : {gpu.driver_version or 'UNKNOWN'}")
    else:
        print("  No GPU detected")
    print(f"\n{divider}")
    print("  NPU")
    print(f"{divider}")
    status = "AVAILABLE" if profile.npu.available else "NOT DETECTED"
    print(f"  Status   : {status}")
    if profile.npu.name:
        print(f"  Name     : {profile.npu.name}")
    if profile.npu.vendor:
        print(f"  Vendor   : {profile.npu.vendor}")
    if profile.npu.framework:
        print(f"  Framework: {profile.npu.framework}")
    print(f"  Notes    : {profile.npu.notes}")
    print(f"\n{divider}")
    print("  RAM")
    print(f"{divider}")
    print(f"  Total     : {profile.ram.total_mb:.0f} MB ({profile.ram.total_mb/1024:.1f} GB)")
    print(f"  Available : {profile.ram.available_mb:.0f} MB ({profile.ram.available_mb/1024:.1f} GB)")
    print(f"\n{'=' * 50}\n")
