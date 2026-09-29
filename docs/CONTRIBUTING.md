# Contributing to SPECTRA

Thank you for your interest in contributing to **SPECTRA: Snapdragon-Powered Private Multimodal AI Workspace**!

## Code of Conduct

All contributors must adhere to respectful and collaborative communication. Zero tolerance for harassment or discrimination.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`:
   ```powershell
   git checkout -b feature/your-feature-name
   ```
2. Ensure all 38 test suite cases pass cleanly:
   ```powershell
   python tests/test_spectra.py
   ```
3. Verify that zero fabricated metrics are introduced in any benchmarking or telemetry modules.
4. Submit a Pull Request with a clear description of the silicon architecture impact and test results.
