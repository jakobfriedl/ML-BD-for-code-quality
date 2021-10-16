# Dataset description

## File name
w3c_test_results_failed.csv

## Description
These are results from selected tests of the w3c [web-platform-tests](https://github.com/w3c/web-platform-tests) repo.

**The web platform tests are as described by the [wpt-project](https://web-platform-tests.org/index.html):** <br>
The web-platform-tests project is a cross-browser test suite for the Web-platform stack. Writing tests in a way that allows them to be run in all browsers gives browser projects confidence that they are shipping software which is compatible with other implementations, and that later implementations will be compatible with their implementations. 

## Number of samples
|                                       |   samples |
|---------------------------------------|-----------|
| Total number of samples               |     68566 |
| Unique samples by ['name', 'message'] |     57892 |
| Unique samples by ['message']         |      8592 |
| Unique samples by ['name']            |     32156 |

## Columns
|    | column        |  description                            |
|----|---------------|-----------------------------------------|
|  0 | name          | the name of the individual test case    |
|  1 | status        | outcome of test run                     |
|  2 | message       | error message of failed test            |
|  3 | group_name    | name of the test case group             |
|  4 | group_status  | outcome of test case group run          |
|  5 | group_message | error message of failed test case group |
|  6 | stack_trace   | stack trace of failed test              |

## Data preview
| name                                                         | status   | message                                                      | group_name                      | group_status   | group_message   | stack_trace   |
|--------------------------------------------------------------|----------|--------------------------------------------------------------|---------------------------------|----------------|-----------------|---------------|
| SensorErrorEvent interface: existence and pro ... ace object | FAIL     | assert_equals: prototype of SensorErrorEvent  ... e code] }" | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent interface: existence and pro ... ype object | FAIL     | assert_equals: prototype of SensorErrorEvent. ... rorEvent]" | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent interface: attribute error                  | FAIL     | assert_own_property: expected property "error" missing       | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent must be primary interface of ... oom!") }); | FAIL     | assert_equals: Unexpected exception when eval ... 2 argu..." | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| Stringification of new SensorErrorEvent({ err ... oom!") }); | FAIL     | assert_equals: Unexpected exception when eval ... 2 argu..." | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| ...                                                          | ...      | ...                                                          | ...                             | ...            | ...             | ...           |
| SensorErrorEvent interface: existence and pro ... ace object | FAIL     | assert_equals: prototype of SensorErrorEvent  ... e code] }" | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent interface: existence and pro ... ype object | FAIL     | assert_equals: prototype of SensorErrorEvent. ... rorEvent]" | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent interface: attribute error                  | FAIL     | assert_own_property: expected property "error" missing       | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| SensorErrorEvent must be primary interface of ... oom!") }); | FAIL     | assert_equals: Unexpected exception when eval ... 2 argu..." | /generic-sensor/idlharness.html | OK             | nan             | nan           |
| Stringification of new SensorErrorEvent({ err ... oom!") }); | FAIL     | assert_equals: Unexpected exception when eval ... 2 argu..." | /generic-sensor/idlharness.html | OK             | nan             | nan           |

## Detailed preview
| name                                                                                                   | message                                                                                                                                                                     |
|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SensorErrorEvent interface: existence and properties of interface object                               | assert_equals: prototype of SensorErrorEvent is not Event expected function "function Event() { [native code] }" but got function "function ErrorEvent() { [native code] }" |
| SensorErrorEvent interface: existence and properties of interface prototype object                     | assert_equals: prototype of SensorErrorEvent.prototype is not Event.prototype expected object "[object Event]" but got object "[object ErrorEvent]"                         |
| SensorErrorEvent interface: attribute error                                                            | assert_own_property: expected property "error" missing                                                                                                                      |
| SensorErrorEvent must be primary interface of new SensorErrorEvent({ error: new TypeError("Boom!") }); | assert_equals: Unexpected exception when evaluating object expected null but got object "TypeError: Failed to construct 'SensorErrorEvent': 2 argu..."                      |
| Stringification of new SensorErrorEvent({ error: new TypeError("Boom!") });                            | assert_equals: Unexpected exception when evaluating object expected null but got object "TypeError: Failed to construct 'SensorErrorEvent': 2 argu..."                      |
| ...                                                                                                    | ...                                                                                                                                                                         |
| SensorErrorEvent interface: existence and properties of interface object                               | assert_equals: prototype of SensorErrorEvent is not Event expected function "function Event() { [native code] }" but got function "function ErrorEvent() { [native code] }" |
| SensorErrorEvent interface: existence and properties of interface prototype object                     | assert_equals: prototype of SensorErrorEvent.prototype is not Event.prototype expected object "[object Event]" but got object "[object ErrorEvent]"                         |
| SensorErrorEvent interface: attribute error                                                            | assert_own_property: expected property "error" missing                                                                                                                      |
| SensorErrorEvent must be primary interface of new SensorErrorEvent({ error: new TypeError("Boom!") }); | assert_equals: Unexpected exception when evaluating object expected null but got object "TypeError: Failed to construct 'SensorErrorEvent': 2 argu..."                      |
| Stringification of new SensorErrorEvent({ error: new TypeError("Boom!") });                            | assert_equals: Unexpected exception when evaluating object expected null but got object "TypeError: Failed to construct 'SensorErrorEvent': 2 argu..."                      |