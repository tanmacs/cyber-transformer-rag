"""
Generate comprehensive validation test report for IOC Tokenizer.
This validates that all IOC types are preserved as atomic tokens.
"""

import json
from pathlib import Path
from datetime import datetime
from train_tokenizer import encode_with_atomic_iocs
from tokenizers import Tokenizer
from pretokenizer import pretokenize, PATTERNS


def generate_validation_report():
    """Generate a comprehensive validation report."""
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Test cases for each IOC type
    test_suite = {
        "CVE_ID": [
            ("CVE-2024-12345", "Critical vulnerability CVE-2024-12345 patched."),
            ("CVE-2021-1234567", "Long CVE CVE-2021-1234567 discovered."),
            ("CVE-2023-44487", "Multi CVE-2023-44487 and CVE-2022-30190 found."),
        ],
        "IPv4": [
            ("192.168.1.105", "Server 192.168.1.105 compromised."),
            ("0.0.0.0", "Traffic 0.0.0.0 and 255.255.255.255 blocked."),
            ("10.0.0.55", "Host 10.0.0.55 infected."),
        ],
        "IPv6": [
            ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "IPv6 2001:0db8:85a3:0000:0000:8a2e:0370:7334 flagged."),
        ],
        "SHA256": [
            ("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", 
             "Hash 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08 matched."),
        ],
        "MD5": [
            ("d41d8cd98f00b204e9800998ecf8427e", "MD5 d41d8cd98f00b204e9800998ecf8427e found."),
        ],
        "Windows Path": [
            ("C:\\Users\\Admin\\AppData\\Local\\malware.exe", "Payload C:\\Users\\Admin\\AppData\\Local\\malware.exe dropped."),
            ("C:\\Windows\\System32\\cmd.exe", "Backdoor C:\\Windows\\System32\\cmd.exe executed."),
        ],
        "Unix Path": [
            ("/usr/local/bin/sshd_monitor", "Rootkit /usr/local/bin/sshd_monitor installed."),
            ("/etc/passwd", "File /etc/passwd compromised."),
        ],
        "Hex String": [
            ("0xdeadbeef", "Shellcode offset 0xdeadbeef identified."),
            ("0x41424344", "Encoding 0x41424344 detected."),
        ],
        "Domain": [
            ("evil-c2.onion", "Beacon evil-c2.onion contacted."),
        ],
    }
    
    # Run validation
    report = {
        "title": "IOC Tokenizer - Atomicity Validation Report",
        "timestamp": datetime.now().isoformat(),
        "tokenizer_config": {
            "model": "Byte-Level BPE",
            "vocab_size": len(tokenizer.get_vocab()),
            "special_tokens": [
                "<|system|>",
                "<|context|>",
                "<|query|>",
                "<|endoftext|>",
            ]
        },
        "test_results": {},
        "summary": {"total": 0, "passed": 0, "failed": 0}
    }
    
    # Test each IOC type
    for ioc_type, test_cases in test_suite.items():
        results = []
        type_passed = 0
        type_failed = 0
        
        for ioc, text in test_cases:
            # Encode with atomic IOC preservation
            result = encode_with_atomic_iocs(tokenizer, text)
            tokens = result["tokens"]
            
            # Check if IOC is preserved
            passed = ioc in tokens
            if passed:
                type_passed += 1
            else:
                type_failed += 1
            
            results.append({
                "ioc": ioc,
                "example": text,
                "preserved": passed,
                "tokens": tokens,
            })
        
        report["test_results"][ioc_type] = {
            "passed": type_passed,
            "failed": type_failed,
            "tests": results
        }
        
        report["summary"]["total"] += len(test_cases)
        report["summary"]["passed"] += type_passed
        report["summary"]["failed"] += type_failed
    
    return report


def print_report(report):
    """Print formatted validation report."""
    
    print("\n" + "=" * 80)
    print(f"  {report['title']}")
    print("=" * 80)
    print(f"\nGenerated: {report['timestamp']}")
    print(f"\nTokenizer Configuration:")
    print(f"  Model: {report['tokenizer_config']['model']}")
    print(f"  Vocabulary Size: {report['tokenizer_config']['vocab_size']:,} tokens")
    print(f"  Special Tokens: {', '.join(report['tokenizer_config']['special_tokens'])}")
    
    print("\n" + "=" * 80)
    print("  TEST RESULTS BY IOC TYPE")
    print("=" * 80)
    
    for ioc_type, results in report["test_results"].items():
        passed = results["passed"]
        failed = results["failed"]
        total = passed + failed
        status = "✓" if failed == 0 else "✗"
        
        print(f"\n[{status}] {ioc_type:20} | {passed}/{total} passed")
        
        for test in results["tests"]:
            mark = "✓" if test["preserved"] else "✗"
            print(f"      {mark} {test['ioc']}")
            print(f"         → {test['example']}")
            if not test["preserved"]:
                print(f"         → Tokens: {test['tokens']}")
    
    print("\n" + "=" * 80)
    print("  OVERALL SUMMARY")
    print("=" * 80)
    total = report["summary"]["total"]
    passed = report["summary"]["passed"]
    failed = report["summary"]["failed"]
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ({pass_rate:.1f}%)")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED - IOCs are preserved as atomic tokens!")
    else:
        print(f"\n✗ {failed} test(s) failed - some IOCs were split across tokens")
    
    print("\n" + "=" * 80)


def save_report_json(report, filename="validation_report.json"):
    """Save report to JSON file."""
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Report saved to: {filename}")


def save_report_html(report, filename="validation_report.html"):
    """Save report as HTML file."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SwiftSafe Tokenizer - Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
            .summary { background: white; padding: 15px; margin: 20px 0; border-left: 4px solid #27ae60; }
            .test-type { background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }
            .pass { color: #27ae60; font-weight: bold; }
            .fail { color: #e74c3c; font-weight: bold; }
            .test-item { padding: 10px; margin: 5px 0; background: #ecf0f1; border-radius: 3px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #34495e; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SwiftSafe Tokenizer - IOC Atomicity Validation Report</h1>
            <p>Generated: """ + report['timestamp'] + """</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Tests</td>
                    <td>""" + str(report['summary']['total']) + """</td>
                </tr>
                <tr>
                    <td>Passed</td>
                    <td class="pass">""" + str(report['summary']['passed']) + """</td>
                </tr>
                <tr>
                    <td>Failed</td>
                    <td class="fail">""" + str(report['summary']['failed']) + """</td>
                </tr>
                <tr>
                    <td>Pass Rate</td>
                    <td class="pass">""" + f"{report['summary']['passed']/report['summary']['total']*100:.1f}%" + """</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html)
    print(f"[✓] HTML report saved to: {filename}")


if __name__ == "__main__":
    print("Generating SwiftSafe Tokenizer Validation Report...")
    
    report = generate_validation_report()
    print_report(report)
    
    save_report_json(report)
    save_report_html(report)
    
    print("\n[✓] Validation complete!")
