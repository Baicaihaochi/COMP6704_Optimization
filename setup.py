"""
Initialize the project by creating necessary directories and __init__.py
"""
import os

# Create src directory if it doesn't exist
os.makedirs('src', exist_ok=True)

# Create results directory
os.makedirs('results', exist_ok=True)

# Create __init__.py for src package
init_file = os.path.join('src', '__init__.py')
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('"""Berlin52 TSP Solution Methods"""\n')
    print(f'Created {init_file}')
else:
    print(f'{init_file} already exists')

print('\nProject structure initialized successfully!')
print('\nDirectory structure:')
print('  src/          - Source code')
print('  results/      - Experimental results (will be populated after running)')
print('\nNext steps:')
print('  1. Check dependencies: pip install -r requirements.txt')
print('  2. Run tests: see TESTING.md')
print('  3. Run experiments: cd src && python run_experiments.py')
