import os


def count_files_in_current_directory():
    """
    Count the number of files in the current directory
    """
    try:
        # List all files in the current directory
        files = [f for f in os.listdir(".") if os.path.isfile(f)]
        count = len(files)

        print(f"✅ Found {count} files in current directory")

        # Optionally list the files
        if count > 0:
            print("Files found:")
            for file in files:
                print(f"  - {file}")

        return count

    except PermissionError:
        print("❌ Permission denied when accessing current directory")
        return 0
    except Exception as e:
        print(f"❌ Error while counting files: {e}")
        return 0


# Main execution
if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    file_count = count_files_in_current_directory()
    print(f"\n📊 Total files in current directory: {file_count}")
