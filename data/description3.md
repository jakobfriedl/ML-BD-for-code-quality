# Dataset description

## File name
monkey_data_stack_trace.csv

## Description
This dataset contains errors and stack traces from a multitude of Android Apps collected via 
Android Studio Monkey tests. <br>
**Github:** <br>
https://github.com/tingsu/DroidDefects/tree/master/ground-truth-cases/Dataset_crashanalysis <br>
https://crashanalysis.github.io/index.html <br>
<br>
**Literature:** <br>
[1]L. Fan u.a., „Large-Scale Analysis of Framework-Specific Exceptions in Android Apps“, 
Proceedings of the 40th International Conference on Software Engineering, S. 408–419, Mai 2018, 
doi: 10.1145/3180155.3180222. <br>
[2]T. Su u.a., „Why My App Crashes Understanding and Benchmarking Framework-specific Exceptions 
of Android apps“, IEEE Trans. Software Eng., S. 1–1, 2020, doi: 10.1109/TSE.2020.3013438.'

## Number of samples
|                                                                 |   samples |
|-----------------------------------------------------------------|-----------|
| Total number of samples                                         |      3945 |
| Unique samples by ['Exception name', 'Pkg name', 'Stack trace'] |      3756 |
| Unique samples by ['Stack trace']                               |      3746 |
| Unique samples by ['Exception name']                            |        74 |

## Columns
|    | column          |  description                                    |
|----|-----------------|-------------------------------------------------|
|  0 | Folder          | Folder of stack trace file in original data set |
|  1 | Pkg name        | Application package name                        |
|  2 | Version name    | Application version number                      |
|  3 | Tool            | Testing Tool                                    |
|  4 | Bug report file | File name of stack trace file                   |
|  5 | Exception name  | Name of raised exception                        |
|  6 | Category        | Category of exception source (framework or app) |
|  7 | Stack trace     | Stack trace of failed test                      |

## Data preview
| Folder                      | Pkg name              | Version name   | Tool   | Bug report file   | Exception name                            | Category   | Stack trace                                                  |
|-----------------------------|-----------------------|----------------|--------|-------------------|-------------------------------------------|------------|--------------------------------------------------------------|
| net.fred.feedex_1.8.1       | net.fred.feedex       | 1.8.1          | Monkey | bugreport.fsm.25  | java.lang.IllegalArgumentException        | app        | //this is an auto-generated bug report//packa ... .java:234) |
| acr.browser.lightning_4.4.1 | acr.browser.lightning | 4.4.1          | Monkey | bugreport.fsm.1   | android.content.ActivityNotFoundException | framework  | //this is an auto-generated bug report//packa ... ve Method) |
| acr.browser.lightning_4.4.2 | acr.browser.lightning | 4.4.2          | Monkey | bugreport.fsm.1   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... ve Method) |
| am.ed.importcontacts_1.3.1  | am.ed.importcontacts  | 1.3.1          | Monkey | bugreport.fsm.1   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... ve Method) |
| am.ed.importcontacts_1.3.1  | am.ed.importcontacts  | 1.3.1          | Monkey | bugreport.fsm.6   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... java:2189) |
| ...                         | ...                   | ...            | ...    | ...               | ...                                       | ...        | ...                                                          |
| net.fred.feedex_1.8.1       | net.fred.feedex       | 1.8.1          | Monkey | bugreport.fsm.25  | java.lang.IllegalArgumentException        | app        | //this is an auto-generated bug report//packa ... .java:234) |
| acr.browser.lightning_4.4.1 | acr.browser.lightning | 4.4.1          | Monkey | bugreport.fsm.1   | android.content.ActivityNotFoundException | framework  | //this is an auto-generated bug report//packa ... ve Method) |
| acr.browser.lightning_4.4.2 | acr.browser.lightning | 4.4.2          | Monkey | bugreport.fsm.1   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... ve Method) |
| am.ed.importcontacts_1.3.1  | am.ed.importcontacts  | 1.3.1          | Monkey | bugreport.fsm.1   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... ve Method) |
| am.ed.importcontacts_1.3.1  | am.ed.importcontacts  | 1.3.1          | Monkey | bugreport.fsm.6   | java.lang.NullPointerException            | app        | //this is an auto-generated bug report//packa ... java:2189) |

## Detailed preview
| Exception name                            | Stack trace                                                                                                                                                                                              |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| java.lang.IllegalArgumentException        | //this is an auto-generated bug report//package name: net.fred.feedex//version: 1.8.1//appro_time: 153 java.lang.RuntimeException: An error occured while executing doInBackground() at a ... .java:234) |
| android.content.ActivityNotFoundException | //this is an auto-generated bug report//package name: acr.browser.lightning//version: 4.4.1//appro_time: 149 android.content.ActivityNotFoundException: No Activity found to handle Inten ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: acr.browser.lightning//version: 4.4.2//appro_time: 103 java.lang.NullPointerException at acr.browser.lightning.fragment.p.a(Unknown ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: am.ed.importcontacts//version: 1.3.1//appro_time: 42 java.lang.NullPointerException at am.ed.importcontacts.Doit$2.onClick(Doit.jav ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: am.ed.importcontacts//version: 1.3.1//appro_time: 47 java.lang.RuntimeException: Unable to start activity ComponentInfo{am.ed.impor ... java:2189) |
| ...                                       | ...                                                                                                                                                                                                      |
| java.lang.IllegalArgumentException        | //this is an auto-generated bug report//package name: net.fred.feedex//version: 1.8.1//appro_time: 153 java.lang.RuntimeException: An error occured while executing doInBackground() at a ... .java:234) |
| android.content.ActivityNotFoundException | //this is an auto-generated bug report//package name: acr.browser.lightning//version: 4.4.1//appro_time: 149 android.content.ActivityNotFoundException: No Activity found to handle Inten ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: acr.browser.lightning//version: 4.4.2//appro_time: 103 java.lang.NullPointerException at acr.browser.lightning.fragment.p.a(Unknown ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: am.ed.importcontacts//version: 1.3.1//appro_time: 42 java.lang.NullPointerException at am.ed.importcontacts.Doit$2.onClick(Doit.jav ... ve Method) |
| java.lang.NullPointerException            | //this is an auto-generated bug report//package name: am.ed.importcontacts//version: 1.3.1//appro_time: 47 java.lang.RuntimeException: Unable to start activity ComponentInfo{am.ed.impor ... java:2189) |