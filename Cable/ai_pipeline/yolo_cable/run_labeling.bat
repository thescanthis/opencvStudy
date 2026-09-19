@echo off
REM X-AnyLabeling 실행 (전용 가상환경, PyQt6 6.7.1)
REM 이 파일을 더블클릭하면 라벨링 프로그램이 뜹니다.
"C:\Users\Administrator\anylabel_env\Scripts\python.exe" -c "from anylabeling.app import main; main()"
pause
