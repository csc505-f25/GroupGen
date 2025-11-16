# Setup Guide: Virtual Environment and IDE Configuration

## The Problem

Your packages (pandas, numpy) are installed in the virtual environment, but:
1. The virtual environment is not activated when you run Python
2. Your IDE might not be using the virtual environment's Python interpreter

## Solution

### Step 1: Activate Virtual Environment in Terminal

**PowerShell (recommended):**
```powershell
cd C:\Users\labuser.DESKTOP-3O6S6S6\Documents\GroupGen
.\venv\Scripts\Activate.ps1
```

**Command Prompt (CMD):**
```cmd
cd C:\Users\labuser.DESKTOP-3O6S6S6\Documents\GroupGen
venv\Scripts\activate.bat
```

**After activation, you should see `(venv)` in your terminal prompt.**

### Step 2: Verify Packages Work

After activating, test:
```bash
python -c "import pandas; import numpy; print('✅ Packages work!')"
```

### Step 3: Configure Your IDE

#### For VS Code / Cursor:

1. **Open Command Palette**: `Ctrl+Shift+P`
2. **Select Python Interpreter**: Type "Python: Select Interpreter"
3. **Choose the virtual environment interpreter**:
   - Look for: `.\venv\Scripts\python.exe`
   - Or: `Python 3.x.x ('venv': venv) ./venv/Scripts/python.exe`

4. **Verify in terminal** (bottom panel):
   - The terminal should show `(venv)` in the prompt
   - If not, click the "+" to create new terminal, or run activation command

```

### Step 4: Run Your Code

**Option A: Run from activated terminal:**
```bash
# Activate venv first
.\venv\Scripts\Activate.ps1

# Then run your script
python backend/main.py
# or
python test_step_by_step.py
```

**Option B: Use the venv Python directly:**
```bash
.\venv\Scripts\python.exe backend/main.py
.\venv\Scripts\python.exe test_step_by_step.py
```

## Quick Test

Create a test file `test_imports.py`:
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

print("✅ All imports successful!")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
```

Run it:
```bash
.\venv\Scripts\python.exe test_imports.py
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
1. Make sure virtual environment is activated (you see `(venv)` in terminal)
2. Check Python interpreter: `python -c "import sys; print(sys.executable)"`
   - Should show: `...\GroupGen\venv\Scripts\python.exe`
3. If not, activate venv or configure IDE to use venv interpreter

### Issue: IDE still shows import errors

**Solution:**
1. Restart your IDE after configuring the interpreter
2. Reload the window in VS Code/Cursor: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Make sure the IDE is using the venv interpreter (check bottom-right corner in VS Code)

### Issue: Packages not installed

**Solution:**
```bash
# Activate venv first
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

## Current Package Status

✅ **Installed:**
- pandas (2.3.3)
- numpy (2.3.4)
- scikit-learn (1.7.2)
- scipy (1.16.3)

❌ **Not installed (optional):**
- scikit-learn-extra (requires C++ build tools - optional, you can implement K-Medoids yourself)

## Note About scikit-learn-extra

The `scikit-learn-extra` package failed to install because it requires C++ build tools. This is **optional** - you can:
1. Use `sklearn.cluster.KMeans` instead (works fine for your use case)
2. Implement K-Medoids yourself (the skeleton code shows how)
3. Install C++ build tools if you really need scikit-learn-extra

For now, you can use K-Means clustering, which works perfectly for your needs!

