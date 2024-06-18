from cx_Freeze import setup, Executable

setup(
    name="covid19",
    version="1.0",
    description="Your application description",
    executables=[Executable("covid19.py")],
    options={
        'build_exe': {
            'includes': ['torch', 'torchvision'],
        }
    }
)
