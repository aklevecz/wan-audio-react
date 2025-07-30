"""
Shared filename sanitization utilities for consistent directory naming.
Used by both backend and frontend (via API) to ensure consistent filesystem paths.
"""

import re
import unicodedata
from pathlib import Path


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """
    Sanitize a filename to be filesystem-safe while maintaining readability.
    
    Args:
        filename: Original filename/project name
        max_length: Maximum length for sanitized name
        
    Returns:
        Sanitized filename safe for use as directory name
        
    Examples:
        "Mount Kimbie & Micachu - Marilyn (Palms Trax Remix)" 
        -> "mount_kimbie_and_micachu_marilyn_palms_trax_remix"
        
        "Artist/Song [2024] (Remix)" 
        -> "artist_song_2024_remix"
    """
    if not filename:
        return "untitled"
    
    # Step 1: Normalize unicode characters (handle accents, etc.)
    normalized = unicodedata.normalize('NFKD', filename)
    
    # Step 2: Convert to lowercase for consistency
    sanitized = normalized.lower()
    
    # Step 3: Replace special characters with readable equivalents
    replacements = {
        '&': 'and',
        '+': 'plus', 
        '@': 'at',
        '%': 'percent',
        '#': 'number',
        '$': 'dollar',
        '€': 'euro',
        '£': 'pound',
        '¥': 'yen',
    }
    
    for char, replacement in replacements.items():
        sanitized = sanitized.replace(char, f'_{replacement}_')
    
    # Step 4: Remove/replace problematic characters
    # Keep only alphanumeric, spaces, hyphens, underscores, dots
    sanitized = re.sub(r'[^\w\s\-_.]', '_', sanitized)
    
    # Step 5: Clean up whitespace and separators
    # Replace multiple spaces/separators with single underscore
    sanitized = re.sub(r'[\s\-_\.]+', '_', sanitized)
    
    # Step 6: Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Step 7: Ensure we have something left
    if not sanitized:
        sanitized = "untitled"
    
    # Step 8: Truncate to max length, preserving word boundaries when possible
    if len(sanitized) > max_length:
        # Try to cut at word boundary (underscore)
        truncated = sanitized[:max_length]
        last_underscore = truncated.rfind('_')
        
        if last_underscore > max_length * 0.7:  # If underscore is reasonably close to end
            sanitized = truncated[:last_underscore]
        else:
            sanitized = truncated
        
        # Ensure we don't end with underscore after truncation
        sanitized = sanitized.rstrip('_')
    
    return sanitized


def create_session_directory_name(project_name: str, timestamp: str, session_id: str) -> str:
    """
    Create a complete session directory name.
    
    Args:
        project_name: Original project/filename
        timestamp: ISO timestamp (will be formatted to YYYY-MM-DD)
        session_id: Full UUID session ID
        
    Returns:
        Complete directory name
        
    Example:
        "mount_kimbie_and_micachu_marilyn_palms_trax_remix_2025-07-12_14b54c4a"
    """
    sanitized_name = sanitize_filename(project_name)
    
    # Format timestamp to YYYY-MM-DD
    if 'T' in timestamp:
        date_part = timestamp.split('T')[0]  # Extract YYYY-MM-DD from ISO
    else:
        date_part = timestamp
    
    # Use first 8 chars of session ID for uniqueness while keeping readable
    session_short = session_id.replace('-', '')[:8]
    
    return f"{sanitized_name}_{date_part}_{session_short}"


def validate_safe_path(path: str) -> bool:
    """
    Validate that a path is safe (no traversal attempts, etc.)
    
    Args:
        path: Path to validate
        
    Returns:
        True if path appears safe
    """
    # Check for path traversal attempts
    if '..' in path:
        return False
    
    # Check for absolute paths
    if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
        return False
    
    # Check for null bytes
    if '\x00' in path:
        return False
    
    return True


# Test cases for validation
if __name__ == "__main__":
    test_cases = [
        "Mount Kimbie & Micachu - Marilyn (Palms Trax Remix)",
        "Artist/Song [2024] (Remix)", 
        "Track with emojis & symbols",
        "Very Long Project Name That Exceeds The Maximum Length Limit And Should Be Truncated Properly",
        "Simple Track",
        "",
        "café résumé naïve",
        "../../../etc/passwd",
        "normal_file.mp3"
    ]
    
    print("Filename Sanitization Tests:")
    print("=" * 80)
    
    for test in test_cases:
        sanitized = sanitize_filename(test)
        safe = validate_safe_path(sanitized)
        print(f"'{test}' -> '{sanitized}' (safe: {safe})")
    
    print("\nDirectory Name Tests:")
    print("=" * 80)
    
    session_id = "14b54c4a-c9f7-47bc-8b40-dd19f6c34826"
    timestamp = "2025-07-12T18:50:05.919763"
    
    for test in test_cases[:3]:
        dir_name = create_session_directory_name(test, timestamp, session_id)
        print(f"'{test}' -> '{dir_name}'")