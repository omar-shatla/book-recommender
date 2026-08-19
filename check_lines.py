with open('tagged_descriptions.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

bad_lines = []
for i, line in enumerate(lines):
    if line.strip():
        first_word = line.strip().split()[0].strip('"')
        if not first_word.isdigit():
            bad_lines.append((i+1, line[:100]))

print(f'Found {len(bad_lines)} truly bad lines out of {len(lines)} total')
print('Sample bad lines:')
for num, content in bad_lines[:15]:
    print(f'Line {num}: {content}')
