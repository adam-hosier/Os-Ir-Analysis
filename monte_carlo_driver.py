for i in range(1000):
    try:
        exec(open(r'./calibrationScript.py').read())
        exec(open(r'./NucRadAnalysis.py').read())
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        print(error_class, detail)
        print(Exception)
