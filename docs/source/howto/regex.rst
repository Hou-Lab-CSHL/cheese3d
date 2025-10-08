Regex guide
===========

To identify source video data and ephys data, Cheese3D uses `"regular expressions" <https://www.regular-expressions.info/quickstart.html>`__. These are data strings that specify a pattern to search for in filenames. Though powerful, they can be daunting for beginners. Cheese3D has several features to make this process easier.

What is a regular expression?
-----------------------------

A **regular expression** (regex) is like a search pattern that can match multiple similar text strings. Think of it as a template that describes what your filenames look like.

For example, if your video files are named like:

* ``mouse1_cal_TL_001.avi``
* ``mouse1_behavior_TR_002.avi``
* ``mouse2_cal_BC_001.avi``

You could describe this pattern as:

1. starts with anything: ``.*``,
2. then underscore: ``_``,
3. then type of video: ``[behavior|cal]``,
4. then underscore: ``_``,
5. then camera view: ``[TL|TR|BC|TC|L|R]``,
6. then underscore: ``_``,
7. then numbers: ``\d+``,
8. then ".avi": ``\.avi``

Putting it all together, we get the regex: ``.*_[behavior|cal]_[TL|TR|BC|TC|L|R]_\d+\.avi``. For the rest of this guide, we will explain how to understand and build such expressions in more detail. For a more comprehesive explanation, see `this guide <https://www.regular-expressions.info/quickstart.html>`__.

Common regex patterns
---------------------

Here are the most common regex symbols you'll encounter:

.. list-table:: Regex pattern cheatsheet
    :header-rows: 1

    * - Pattern
      - Meaning
      - Example use
      - Example matches
    * - ``.``
      - Any single character
      - ``a.c``
      - abc, a1c, a-c
    * - ``*``
      - Zero or more of previous pattern
      - ``ab*c``
      - ac, abc, abbc
    * - ``+``
      - One or more of previous pattern
      - ``ab+c``
      - abc, abbc (not ac)
    * - ``[abc]``
      - Any of the characters a, b, or c
      - ``[TL]R``
      - TR, LR
    * - ``[^abc]``
      - Any character except a, b, or c
      - ``[^_]+``
      - mouse1, cal (not _)
    * - ``|``
      - OR operator
      - ``TL|TR|BC``
      - TL, TR, or BC
    * - ``\.``
      - Literal dot (escaped)
      - ``file\.avi``
      - file.avi
    * - ``\d``
      - Any digit 0-9
      - ``\d+``
      - 1, 123, 007
    * - ``\w``
      - Any word character (letter, digit, or underscore)
      - ``\w+``
      - mouse1, cal, TL
    * - ``.*``
      - Any characters (wildcard)
      - ``.*\.avi``
      - anything ending in .avi

Named groups in Python regex
-----------------------------

Python regex supports **named groups** which let you capture and reference specific parts of the match by name instead of position.

**Syntax**: ``(?P<name>pattern)`` creates a named group called "name"

**Example:**
``(?P<mouse>mouse\d+)_(?P<type>\w+)_(?P<view>TL|TR|BC)_(?P<session>\d+)\.avi``

This would match ``mouse1_cal_TL_001.avi`` and capture:

.. list-table::
    :header-rows: 1

    * - Key
      - Pattern
      - Matched value
    * - ``mouse``
      - ``mouse\d+``
      - ``mouse1``
    * - ``type``
      - ``\w+``
      - ``cal``
    * - ``view``
      - ``TL|TR|BC``
      - ``TL``
    * - ``session``
      - ``\d+``
      - ``001``

Cheese3D configuration dictionary format
----------------------------------------

Cheese3D allows you to define regex patterns using a more readable dictionary format instead of writing complex regex strings directly.

Dictionary structure
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   video_regex:
     _path_: ".*_{{type}}_{{view}}_{{session}}.*\.avi"
     type: "[^_]+"
     view: "TL|TR|L|R|TC|BC"
     session: "\d+"

**Required Keys:**

* ``_path_``: The main filename regex pattern with ``{{placeholders}}`` for groups
* ``view``: Camera view pattern (required for multi-camera setups)

**Optional Keys:** any additional groups you want to capture (``type``, ``session``, etc.)

Internally, Cheese3D will build the corresponding regex string for you. For example, the transformation looks something like this:

.. code-block:: python

   # Dictionary input:
   {
     "_path_": ".*_{{type}}_{{view}}.*\.avi",
     "type": "[^_]+",
     "view": "TL|TR|L|R|TC|BC"
   }

   # Becomes Python regex:
   ".*_(?P<type>[^_]+)_(?P<view>TL|TR|L|R|TC|BC).*\.avi"

Using regex groups in Cheese3D configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once defined, the regex groups can be used throughout your configuration:

**In sessions** (see :ref:`reference/configuration:Recording options`):

.. code-block:: yaml

   sessions:
   - name: session1
     type: behavior  # Filters files where type group = "behavior"
   - name: session2
     type: cal       # Filters files where type group = "cal"

**In calibration** (see :ref:`reference/configuration:Calibration options`):

.. code-block:: yaml

   calibration:
     type: cal  # Uses files where type group = "cal"

This allows you to organize different types of videos (calibration vs. behavior) and different sessions within the same directory structure.
