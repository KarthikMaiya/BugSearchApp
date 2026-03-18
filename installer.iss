#define MyAppName "BugSearchApp"
#define MyAppVersion "1.0"
#define MyAppExeName "BugSearchApp.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\BugSearchApp
DefaultGroupName=BugSearchApp
OutputDir=installer_output
OutputBaseFilename=BugSearchApp_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\BugSearchApp\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\BugSearchApp"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\BugSearchApp"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch BugSearchApp"; Flags: nowait postinstall skipifsilent
