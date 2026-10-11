import os

# Nayi file ka naam jisme saara code merge hoga
output_file = "all_code_for_claude.txt"

# Sirf in extensions wali files ko hi read karna hai
allowed_extensions = ('.py', '.txt', '.md')

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir('.'):
        # Check karein ki file text/code file hai aur output file khud nahi hai
        if filename.endswith(allowed_extensions) and filename != output_file and filename != 'merge_code.py':
            
            # Har file ke code se pehle uska naam likhne ke liye formatting
            outfile.write(f"\n\n{'='*50}\n")
            outfile.write(f"File: {filename}\n")
            outfile.write(f"{'='*50}\n\n") # <-- Yahan correction ki gayi hai
            
            try:
                with open(filename, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Error reading file: {e}\n")

print(f"Sabhi files successfully '{output_file}' mein merge ho gayi hain!")