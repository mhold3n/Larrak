#!/usr/bin/env python3
"""
Silent Fallback Audit Script

Systematically searches the codebase for silent fallback patterns:
1. Exception handling that silently returns defaults
2. Fallback comments and markers
3. Conditional fallback logic
4. Default value assignments on failure paths
"""

import os
import re
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class FallbackFinding:
    """Represents a single fallback finding."""
    file_path: str
    line_number: int
    line_content: str
    pattern_type: str  # exception_handling, comment_marker, conditional, default_value
    severity: str = "UNKNOWN"  # CRITICAL, HIGH, MEDIUM, LOW, LEGITIMATE
    context: str = ""
    impact: str = ""
    documented: bool = False
    fix_status: str = "NEEDS_REVIEW"  # FIXED, NEEDS_FIX, LEGITIMATE, NEEDS_REVIEW

@dataclass
class AuditReport:
    """Complete audit report."""
    findings: list[FallbackFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    
    def add_finding(self, finding: FallbackFinding) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)
    
    def generate_summary(self) -> None:
        """Generate summary statistics."""
        self.summary = {
            "total_findings": len(self.findings),
            "by_pattern_type": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_fix_status": defaultdict(int),
            "files_affected": len(set(f.file_path for f in self.findings)),
        }
        
        for finding in self.findings:
            self.summary["by_pattern_type"][finding.pattern_type] += 1
            self.summary["by_severity"][finding.severity] += 1
            self.summary["by_fix_status"][finding.fix_status] += 1

def find_exception_handling_patterns(file_path: Path, report: AuditReport) -> None:
    """Find exception handling patterns that silently return defaults."""
    patterns = [
        (r'except\s*:\s*pass', 'bare_except_pass'),
        (r'except\s+Exception\s*:\s*pass', 'exception_pass'),
        (r'except\s+.*:\s*pass', 'specific_except_pass'),
        (r'except\s+.*:\s*return\s+None', 'return_none'),
        (r'except\s+.*:\s*return\s+""', 'return_empty_string'),
        (r'except\s+.*:\s*return\s+\[\]', 'return_empty_list'),
        (r'except\s+.*:\s*return\s+\{\}', 'return_empty_dict'),
        (r'except\s+.*:\s*return\s+0', 'return_zero'),
        (r'except\s+.*:\s*return\s+False', 'return_false'),
        (r'except\s+.*:\s*continue', 'except_continue'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for pattern, pattern_name in patterns:
                if re.search(pattern, line):
                    finding = FallbackFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip(),
                        pattern_type='exception_handling',
                        context=_get_context(lines, line_num),
                    )
                    report.add_finding(finding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)

def find_fallback_comments(file_path: Path, report: AuditReport) -> None:
    """Find fallback-related comments."""
    patterns = [
        (r'#.*fallback', 'fallback_comment'),
        (r'#.*TODO', 'todo_comment'),
        (r'#.*FIXME', 'fixme_comment'),
        (r'#.*XXX', 'xxx_comment'),
        (r'#.*HACK', 'hack_comment'),
        (r'#.*TEMP', 'temp_comment'),
        (r'#.*PLACEHOLDER', 'placeholder_comment'),
        (r'#.*STUB', 'stub_comment'),
        (r'#.*temporary', 'temporary_comment'),
        (r'#.*workaround', 'workaround_comment'),
        (r'#.*legacy', 'legacy_comment'),
        (r'#.*deprecated', 'deprecated_comment'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for pattern, pattern_name in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    finding = FallbackFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip(),
                        pattern_type='comment_marker',
                        context=_get_context(lines, line_num),
                    )
                    report.add_finding(finding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)

def find_conditional_fallbacks(file_path: Path, report: AuditReport) -> None:
    """Find conditional fallback logic."""
    patterns = [
        (r'if\s+not\s+\w+.*:\s*use_fallback', 'if_not_use_fallback'),
        (r'if\s+not\s+\w+.*:\s*\w+\(\)\s*#.*fallback', 'if_not_fallback_call'),
        (r'try:.*except.*:\s*\w+\(\)\s*#.*(deprecated|legacy|old)', 'try_except_legacy'),
        (r'if\s+.*:\s*use_legacy\(\)', 'if_use_legacy'),
        (r'if\s+.*:\s*use_modern\(\)', 'if_use_modern'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for line_num, line in enumerate(lines, 1):
            for pattern, pattern_name in patterns:
                if re.search(pattern, line, re.IGNORECASE | re.MULTILINE):
                    finding = FallbackFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip(),
                        pattern_type='conditional',
                        context=_get_context(lines, line_num),
                    )
                    report.add_finding(finding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)

def find_default_assignments(file_path: Path, report: AuditReport) -> None:
    """Find default value assignments on failure paths."""
    patterns = [
        (r'\w+\s*=\s*\w+\(\)\s+or\s+\w+', 'or_default'),
        (r'\w+\s*=\s*\w+\(\)\s+if\s+\w+\(\)\s+else\s+\w+', 'ternary_default'),
        (r'except.*:\s*\w+\s*=\s*\d+\.?\d*', 'except_assign_numeric'),
        (r'except.*:\s*\w+\s*=\s*["\'].*["\']', 'except_assign_string'),
        (r'except.*:\s*\w+\s*=\s*\[\]', 'except_assign_list'),
        (r'except.*:\s*\w+\s*=\s*\{\}', 'except_assign_dict'),
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            for pattern, pattern_name in patterns:
                if re.search(pattern, line):
                    finding = FallbackFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.strip(),
                        pattern_type='default_value',
                        context=_get_context(lines, line_num),
                    )
                    report.add_finding(finding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)

def _get_context(lines: list[str], line_num: int, context_lines: int = 3) -> str:
    """Get context around a line."""
    start = max(0, line_num - context_lines - 1)
    end = min(len(lines), line_num + context_lines)
    context = lines[start:end]
    return '\n'.join(f"{start + i + 1:4d}: {line.rstrip()}" for i, line in enumerate(context))

def audit_codebase(root_dir: Path = Path('campro')) -> AuditReport:
    """Audit the entire codebase for silent fallbacks."""
    report = AuditReport()
    
    # Find all Python files relative to project root
    project_root = Path(__file__).parent.parent
    root_path = project_root / root_dir
    
    if not root_path.exists():
        print(f"Error: {root_path} does not exist", file=sys.stderr)
        return report
    
    python_files = list(root_path.rglob('*.py'))
    
    print(f"Auditing {len(python_files)} Python files...", file=sys.stderr)
    
    for file_path in python_files:
        # Skip test files and __pycache__
        if 'test' in str(file_path) or '__pycache__' in str(file_path):
            continue
            
        find_exception_handling_patterns(file_path, report)
        find_fallback_comments(file_path, report)
        find_conditional_fallbacks(file_path, report)
        find_default_assignments(file_path, report)
    
    report.generate_summary()
    return report

def print_report(report: AuditReport, output_file: Path | None = None) -> None:
    """Print the audit report."""
    output = output_file.open('w') if output_file else sys.stdout
    
    print("# Silent Fallback Audit Report\n", file=output)
    print(f"**Total Findings**: {report.summary.get('total_findings', 0)}\n", file=output)
    print(f"**Files Affected**: {report.summary.get('files_affected', 0)}\n", file=output)
    
    print("## Summary by Pattern Type\n", file=output)
    for pattern_type, count in report.summary.get('by_pattern_type', {}).items():
        print(f"- {pattern_type}: {count}", file=output)
    
    print("\n## Summary by Severity\n", file=output)
    for severity, count in report.summary.get('by_severity', {}).items():
        print(f"- {severity}: {count}", file=output)
    
    print("\n## Detailed Findings\n", file=output)
    
    # Group by file
    by_file = defaultdict(list)
    for finding in report.findings:
        by_file[finding.file_path].append(finding)
    
    for file_path in sorted(by_file.keys()):
        print(f"\n### {file_path}\n", file=output)
        for finding in sorted(by_file[file_path], key=lambda f: f.line_number):
            print(f"**Line {finding.line_number}** ({finding.pattern_type}):", file=output)
            print(f"```python", file=output)
            print(f"{finding.line_content}", file=output)
            print(f"```", file=output)
            if finding.context:
                print(f"Context:\n```python", file=output)
                print(f"{finding.context}", file=output)
                print(f"```", file=output)
            print("", file=output)
    
    if output_file:
        output.close()

if __name__ == '__main__':
    root = Path(__file__).parent.parent
    os.chdir(root)
    
    report = audit_codebase()
    
    output_path = root / 'docs' / 'silent_fallback_audit_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print_report(report, output_path)
    print(f"\nAudit complete. Report saved to {output_path}", file=sys.stderr)

