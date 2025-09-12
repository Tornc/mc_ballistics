@ECHO OFF

IF      "%~1"==""           CALL :run       & GOTO :end
IF /I   "%~1"== "clean"     CALL :clean     & GOTO :end
IF /I   "%~1"== "build"     CALL :build     & GOTO :end

CALL :unknown & GOTO :end

:: === Functions ===

:end
    EXIT /B

:unknown
    ECHO Unknown parameter: %1
    GOTO :EOF

:run
    streamlit run app.py
    GOTO :EOF

:build
    echo "not implemented yet"
    GOTO :EOF

:clean
    echo "not implemented yet"
    GOTO :EOF