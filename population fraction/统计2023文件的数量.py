import os

def count_xlsx_files_in_folder(folder_path="."):
    """
    Count the number of xlsx files in a specified folder

    Args:
        folder_path (str): Path to the folder to check, defaults to current directory

    Returns:
        int: Number of xlsx files found
    """
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Folder `{folder_path}` does not exist")
            return 0

        # List all files in the directory and count those ending with .xlsx
        xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        count = len(xlsx_files)

        print(f"✅ Found {count} .xlsx files in `{folder_path}`")

        # Optionally list the files
        if count > 0:
            print("Files found:")
            for file in xlsx_files:
                print(f"  - {file}")

        return count

    except PermissionError:
        print(f"❌ Permission denied when accessing `{folder_path}`")
        return 0
    except Exception as e:
        print(f"❌ Error while counting files: {e}")
        return 0

# Main execution
if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    # Count xlsx files in current folder since script is already in population fraction
    xlsx_count = count_xlsx_files_in_folder(".")
    print(f"\n📊 Total .xlsx files in current directory: {xlsx_count}")
