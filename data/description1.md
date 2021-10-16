# Dataset description

## File name
github_issues.csv

## Description
This dataset contains issues and stack traces collected from GitHub. <br>'
**GitHub:** <br>
https://github.com/tingsu/DroidDefects/tree/master/ground-truth-cases/Dataset_crashanalysis <br>
https://crashanalysis.github.io/index.html <br>
<br>
**Literature:** <br>'
[1]L. Fan u.a., „Large-Scale Analysis of Framework-Specific Exceptions in Android Apps“,
Proceedings of the 40th International Conference on Software Engineering, S. 408–419, Mai 2018,
doi: 10.1145/3180155.3180222. <br>'
[2]T. Su u.a., „Why My App Crashes Understanding and Benchmarking Framework-specific Exceptions
of Android apps“, IEEE Trans. Software Eng., S. 1–1, 2020, doi: 10.1109/TSE.2020.3013438.'

## Number of samples
|                                                                 |   samples |
|-----------------------------------------------------------------|-----------|
| Total number of samples                                         |      6683 |
| Unique samples by ['Exception name', 'Pkg name', 'Stack trace'] |      5245 |
| Unique samples by ['Stack trace']                               |      4810 |
| Unique samples by ['Exception name']                            |       255 |

## Columns
|    | column         |  description                      |
|----|----------------|-----------------------------------|
|  0 | Project        | GitHub Project                    |
|  1 | Pkg name       | Application package name          |
|  2 | Issue ID       | Github Issue ID                   |
|  3 | Issue Link     | Link to GitHub issue              |
|  4 | Exception name | Name of raised exception          |
|  5 | #Comment       | Number of comments on Issue       |
|  6 | Open date      | Opening date of issue             |
|  7 | Closed date    | Closing date of issue             |
|  8 | Duration (day) | Number of days the issue was open |
|  9 | Category       | Category of issue source          |
| 10 | Stack trace    | Stack trace of issue              |

## Data preview
| Project                            | Pkg name                        | Issue ID   | Issue Link                                                   | Exception name                                               | #Comment   | Open date            | Closed date          | Duration (day)   | Category    | Stack trace                                                  |
|------------------------------------|---------------------------------|------------|--------------------------------------------------------------|--------------------------------------------------------------|------------|----------------------|----------------------|------------------|-------------|--------------------------------------------------------------|
| wordpress-mobile/WordPress-Android | org.wordpress.android           | 5699       | https://github.com/wordpress-mobile/WordPress ... ssues/5699 | StringIndexOutOfBoundsException, regionLength ... java:1931) | 1          | 2017-04-20T07:55:27Z | 2017-04-28T08:45:08Z | 8.0              | framework   | Fatal Exception: java.lang.RuntimeException:  ... .java:776) |
| connectbot/connectbot              | org.connectbot                  | 493        | https://github.com/connectbot/connectbot/issues/493          | StringIndexOutOfBoundsException, regionLength ... .java:298) | 1          | 2017-02-27T18:39:53Z | 2017-03-02T18:52:29Z | 3.0              | framework   | 02-28 02:29:35.819 27000 27000 AndroidRuntime ... .java:616) |
| trikita/talalarmo                  | trikita.talalarmo               | 15         | https://github.com/trikita/talalarmo/issues/15               | NoSuchMethodError, or its super classes (decl ... e.java:44) | 0          | 2016-10-25T08:19:59Z | 2016-12-25T08:14:52Z | 61.0             | app         | java.lang.NoSuchMethodError: No direct method ... .java:616) |
| zxing/zxing                        | com.google.zxing.client.android | 767        | https://github.com/zxing/zxing/issues/767                    | NoSuchMethodError, or its super classes (decl ... t.java:31) | 1          | 2017-03-07T17:56:56Z | 2017-03-07T19:19:37Z | 0.0              | libcore/lib | java.lang.NoSuchMethodError: No virtual meth ... .java:755)  |
| wordpress-mobile/WordPress-Android | org.wordpress.android           | 5677       | https://github.com/wordpress-mobile/WordPress ... ssues/5677 | InvalidClassException,but expected org.wordpr ... java:2336) | 1          | 2017-04-17T12:24:18Z | nan                  | nan              | framework   | Fatal Exception: java.lang.RuntimeException:  ... java:1120) |
| ...                                | ...                             | ...        | ...                                                          | ...                                                          | ...        | ...                  | ...                  | ...              | ...         | ...                                                          |
| wordpress-mobile/WordPress-Android | org.wordpress.android           | 5699       | https://github.com/wordpress-mobile/WordPress ... ssues/5699 | StringIndexOutOfBoundsException, regionLength ... java:1931) | 1          | 2017-04-20T07:55:27Z | 2017-04-28T08:45:08Z | 8.0              | framework   | Fatal Exception: java.lang.RuntimeException:  ... .java:776) |
| connectbot/connectbot              | org.connectbot                  | 493        | https://github.com/connectbot/connectbot/issues/493          | StringIndexOutOfBoundsException, regionLength ... .java:298) | 1          | 2017-02-27T18:39:53Z | 2017-03-02T18:52:29Z | 3.0              | framework   | 02-28 02:29:35.819 27000 27000 AndroidRuntime ... .java:616) |
| trikita/talalarmo                  | trikita.talalarmo               | 15         | https://github.com/trikita/talalarmo/issues/15               | NoSuchMethodError, or its super classes (decl ... e.java:44) | 0          | 2016-10-25T08:19:59Z | 2016-12-25T08:14:52Z | 61.0             | app         | java.lang.NoSuchMethodError: No direct method ... .java:616) |
| zxing/zxing                        | com.google.zxing.client.android | 767        | https://github.com/zxing/zxing/issues/767                    | NoSuchMethodError, or its super classes (decl ... t.java:31) | 1          | 2017-03-07T17:56:56Z | 2017-03-07T19:19:37Z | 0.0              | libcore/lib | java.lang.NoSuchMethodError: No virtual meth ... .java:755)  |
| wordpress-mobile/WordPress-Android | org.wordpress.android           | 5677       | https://github.com/wordpress-mobile/WordPress ... ssues/5677 | InvalidClassException,but expected org.wordpr ... java:2336) | 1          | 2017-04-17T12:24:18Z | nan                  | nan              | framework   | Fatal Exception: java.lang.RuntimeException:  ... java:1120) |

## Detailed preview
| Exception name                                                                                                                                                                                           | Stack trace                                                                                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| StringIndexOutOfBoundsException, regionLength=-13, at java.lang.String.substring(String.java:1931)                                                                                                       | Fatal Exception: java.lang.RuntimeException: Unable to start service org.wordpress.android.ui.media.services.MediaUploadService@95d5ff7 with Intent { cmp=org.wordpress.android/.ui.media ... .java:776) |
| StringIndexOutOfBoundsException, regionLength=-1, at java.lang.String.startEndAndLength(String.java:298)                                                                                                 | 02-28 02:29:35.819 27000 27000 AndroidRuntime E java.lang.StringIndexOutOfBoundsException: length=16; regionStart=0; regionLength=-102-28 02:29:35.819 27000 27000 AndroidRuntime E at ja ... .java:616) |
| NoSuchMethodError, or its super classes (declaration of 'trikita.talalarmo.ui.-$Lambda$25' appears in /data/app/trikita.talalarmo-2/base.apk), at trikita.talalarmo.ui.Theme.materialIcon(Theme.java:44) | java.lang.NoSuchMethodError: No direct method <init>(Ljava/lang/Object;)V in class Ltrikita/talalarmo/ui/-$Lambda$25; or its super classes (declaration of 'trikita.talalarmo.ui.-$Lambda ... .java:616) |
| NoSuchMethodError, or its super classes (declaration of 'com.google.zxing.integration.android.IntentIntegrator' appears in /data/app/google.zxing.integration.android-2/split_lib_depende ... t.java:31) | java.lang.NoSuchMethodError: No virtual method initiateScan()Landroid/app/AlertDialog; in class Lcom/google/zxing/integration/android/IntentIntegrator; or its super classes (declaratio ... .java:755)  |
| InvalidClassException,but expected org.wordpress.android.fluxc.model.SiteModel: static final long serialVersionUID =7223616871655133993L, at java.io.ObjectInputStream.verifyAndInit(Obje ... java:2336) | Fatal Exception: java.lang.RuntimeException: Unable to start activity ComponentInfo{org.wordpress.android/org.wordpress.android.ui.posts.EditPostActivity}: java.lang.RuntimeException: P ... java:1120) |
| ...                                                                                                                                                                                                      | ...                                                                                                                                                                                                      |
| StringIndexOutOfBoundsException, regionLength=-13, at java.lang.String.substring(String.java:1931)                                                                                                       | Fatal Exception: java.lang.RuntimeException: Unable to start service org.wordpress.android.ui.media.services.MediaUploadService@95d5ff7 with Intent { cmp=org.wordpress.android/.ui.media ... .java:776) |
| StringIndexOutOfBoundsException, regionLength=-1, at java.lang.String.startEndAndLength(String.java:298)                                                                                                 | 02-28 02:29:35.819 27000 27000 AndroidRuntime E java.lang.StringIndexOutOfBoundsException: length=16; regionStart=0; regionLength=-102-28 02:29:35.819 27000 27000 AndroidRuntime E at ja ... .java:616) |
| NoSuchMethodError, or its super classes (declaration of 'trikita.talalarmo.ui.-$Lambda$25' appears in /data/app/trikita.talalarmo-2/base.apk), at trikita.talalarmo.ui.Theme.materialIcon(Theme.java:44) | java.lang.NoSuchMethodError: No direct method <init>(Ljava/lang/Object;)V in class Ltrikita/talalarmo/ui/-$Lambda$25; or its super classes (declaration of 'trikita.talalarmo.ui.-$Lambda ... .java:616) |
| NoSuchMethodError, or its super classes (declaration of 'com.google.zxing.integration.android.IntentIntegrator' appears in /data/app/google.zxing.integration.android-2/split_lib_depende ... t.java:31) | java.lang.NoSuchMethodError: No virtual method initiateScan()Landroid/app/AlertDialog; in class Lcom/google/zxing/integration/android/IntentIntegrator; or its super classes (declaratio ... .java:755)  |
| InvalidClassException,but expected org.wordpress.android.fluxc.model.SiteModel: static final long serialVersionUID =7223616871655133993L, at java.io.ObjectInputStream.verifyAndInit(Obje ... java:2336) | Fatal Exception: java.lang.RuntimeException: Unable to start activity ComponentInfo{org.wordpress.android/org.wordpress.android.ui.posts.EditPostActivity}: java.lang.RuntimeException: P ... java:1120) |