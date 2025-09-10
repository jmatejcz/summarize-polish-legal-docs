import json
import os
import argparse
from pathlib import Path
from typing import List, Optional


def extract_summaries_from_json(
    json_file_path: str,
    output_dir: str,
    filename_filter: Optional[List[str]] = None,
    model_name: str = "",
) -> int:
    """
    Extract generated summaries from a single JSON file and save each to a separate text file.

    Args:
        json_file_path: Path to the input JSON file
        output_dir: Directory to save the summary files
        filename_filter: Optional list of filenames to extract. If None, extracts all.
        model_name: Name of the model (for prefixing output files)

    Returns:
        Number of summaries extracted
    """
    try:
        # Load the JSON data
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract detailed results
        detailed_results = data.get("detailed_results", [])

        if not detailed_results:
            print(f"  No detailed_results found in {json_file_path}")
            return 0

        extracted_count = 0
        processed_files = set()  # Track unique filenames to avoid duplicates

        for result in detailed_results:
            filename = result.get("filename", "")
            generated_summary = result.get("generated_summary", "")

            # Skip if no filename or summary
            if not filename or not generated_summary:
                continue

            # Apply filename filter if provided
            if filename_filter and filename not in filename_filter:
                continue

            # Skip if we've already processed this filename (handles repeat_number cases)
            if filename in processed_files:
                continue

            processed_files.add(filename)

            # Create output filename
            base_name = Path(filename).stem
            if model_name:
                output_filename = f"{model_name}_{base_name}_summary.txt"
            else:
                output_filename = f"{base_name}_summary.txt"

            output_path = Path(output_dir) / output_filename

            # Save the summary to file
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(generated_summary)

                print(f"  ✓ Saved summary for '{filename}' -> '{output_path}'")
                extracted_count += 1

            except Exception as e:
                print(f"  ✗ Error saving summary for '{filename}': {e}")

        return extracted_count

    except FileNotFoundError:
        print(f"  Error: JSON file '{json_file_path}' not found.")
        return 0
    except json.JSONDecodeError as e:
        print(f"  Error: Invalid JSON format in '{json_file_path}' - {e}")
        return 0
    except Exception as e:
        print(f"  Error processing '{json_file_path}': {e}")
        return 0


def process_qlora_directory(
    qlora_dir: str,
    output_dir: str = "summaries",
    filename_filter: Optional[List[str]] = None,
    config_filter: Optional[str] = None,
) -> None:
    """
    Process all model directories in the qlora directory.

    Args:
        qlora_dir: Path to the qlora directory containing model subdirectories
        output_dir: Directory to save the summary files
        filename_filter: Optional list of filenames to extract. If None, extracts all.
        config_filter: Optional configuration name to filter (e.g., 'aggressive', 'conservative')
    """

    qlora_path = Path(qlora_dir)

    if not qlora_path.exists():
        print(f"Error: Directory '{qlora_dir}' not found.")
        return

    if not qlora_path.is_dir():
        print(f"Error: '{qlora_dir}' is not a directory.")
        return

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    total_extracted = 0
    processed_models = 0

    # Look for subdirectories containing test_evaluation_results.json
    for model_dir in qlora_path.iterdir():
        if not model_dir.is_dir():
            continue

        # Apply config filter if specified
        if config_filter:
            model_name_lower = model_dir.name.lower()
            config_filter_lower = config_filter.lower()
            if config_filter_lower not in model_name_lower:
                continue

        json_file = model_dir / "test_evaluation_results.json"

        if not json_file.exists():
            continue

        model_name = model_dir.name
        print(f"\n📁 Processing model: {model_name}")

        count = extract_summaries_from_json(
            str(json_file), output_dir, filename_filter, model_name
        )

        total_extracted += count
        processed_models += 1

        print(f"  Extracted {count} summaries from {model_name}")

    if processed_models == 0:
        if config_filter:
            print(
                f"\nNo model directories containing '{config_filter}' with 'test_evaluation_results.json' found in '{qlora_dir}'"
            )
        else:
            print(
                f"\nNo model directories with 'test_evaluation_results.json' found in '{qlora_dir}'"
            )

        # Show available directories
        subdirs = [d.name for d in qlora_path.iterdir() if d.is_dir()]
        if subdirs:
            print(f"Available subdirectories:")
            for d in subdirs:
                json_path = qlora_path / d / "test_evaluation_results.json"
                status = "✓" if json_path.exists() else "✗"
                config_match = ""
                if config_filter:
                    config_match = (
                        " (matches config)"
                        if config_filter.lower() in d.lower()
                        else " (no config match)"
                    )
                print(f"  {status} {d}{config_match}")
    else:
        config_info = f" (config: {config_filter})" if config_filter else ""
        print(
            f"\n🎉 Completed! Processed {processed_models} models{config_info} and extracted {total_extracted} summaries total."
        )
        print(f"All summaries saved to '{output_dir}' directory.")


def process_single_json(
    json_file: str,
    output_dir: str = "summaries",
    filename_filter: Optional[List[str]] = None,
) -> None:
    """
    Process a single JSON file.

    Args:
        json_file: Path to the JSON file
        output_dir: Directory to save the summary files
        filename_filter: Optional list of filenames to extract. If None, extracts all.
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

    print(f"📄 Processing single JSON file: {json_file}")

    count = extract_summaries_from_json(json_file, output_dir, filename_filter)

    print(f"\n🎉 Completed! Extracted {count} summaries to '{output_dir}' directory.")

    # Show available filenames if filter was used but no matches found
    if filename_filter and count == 0:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            detailed_results = data.get("detailed_results", [])
            available_files = [
                result.get("filename", "")
                for result in detailed_results
                if result.get("filename")
            ]
            unique_files = sorted(set(available_files))
            if unique_files:
                print(f"\nNo matches found for the specified filenames.")
                print(f"Available filenames in the JSON file:")
                for f in unique_files:
                    print(f"  - {f}")
        except:
            pass


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract generated summaries from JSON evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models in qlora directory
  python extract_summaries.py /path/to/qlora
  
  # Process only aggressive configuration models
  python extract_summaries.py /path/to/qlora --config aggressive
  
  # Process only conservative models with custom output directory
  python extract_summaries.py /path/to/qlora --config conservative --output-dir conservative_summaries
  
  # Process single JSON file
  python extract_summaries.py results.json
  
  # Extract to specific directory
  python extract_summaries.py /path/to/qlora --output-dir my_summaries
  
  # Extract only specific files from aggressive models
  python extract_summaries.py /path/to/qlora --config aggressive --files "VIII_GNc_338423_pozew_o_zaplate.txt" "I_C_220223_pozew_o_zaplate.txt"
        """,
    )

    parser.add_argument(
        "input_path",
        help="Path to qlora directory (containing model subdirs) or single JSON file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="summaries",
        help="Output directory for summary files (default: summaries)",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="*",
        help="List of specific filenames to extract (optional)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help='Configuration type to filter (e.g., "aggressive", "conservative", "moderate")',
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if input_path.is_dir():
        # Process directory
        process_qlora_directory(
            qlora_dir=args.input_path,
            output_dir=args.output_dir,
            filename_filter=args.files,
            config_filter=args.config,
        )
    elif input_path.is_file() and input_path.suffix == ".json":
        # Process single JSON file
        process_single_json(
            json_file=args.input_path,
            output_dir=args.output_dir,
            filename_filter=args.files,
        )
    else:
        print(f"Error: '{args.input_path}' is not a valid directory or JSON file.")
        return 1


if __name__ == "__main__":
    main()
