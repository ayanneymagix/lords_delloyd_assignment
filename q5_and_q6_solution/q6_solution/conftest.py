import pytest

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write a custom summary report after tests finish."""
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    with open("test_report.txt", "w", encoding="utf-8") as f:
        f.write("==== Automated License Plate Similarity Tests ====\n\n")
        f.write(f"Total tests run: {total}\n")
        f.write(f"✅ Passed: {passed}\n")
        f.write(f"❌ Failed: {failed}\n")
        f.write(f"⚠️ Skipped: {skipped}\n\n")

        if "failed" in terminalreporter.stats:
            f.write("---- Failed Test Details ----\n")
            for rep in terminalreporter.stats["failed"]:
                f.write(f"{rep.nodeid}\n")
                f.write(f"{rep.longrepr}\n\n")

    print("\n📄 Report saved to test_report.txt\n")
