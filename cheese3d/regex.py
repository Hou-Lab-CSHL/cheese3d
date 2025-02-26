RECORDING_DATE_REGEX = r"([0-9]{8})"
RECORDING_MOUSE_REGEX = r"([BRMHCDEF][0-9]+|test|all)"
RECORDING_EXP_REGEX = r"([^_]+)"
RECORDING_START_REGEX = "_".join([RECORDING_DATE_REGEX,
                                  RECORDING_MOUSE_REGEX,
                                  RECORDING_EXP_REGEX])
RECORDING_COND_REGEX = r"([^_]+)"
RECORDING_RUN_REGEX = r"(\d{3})"
RECORDING_END_REGEX = r"([0-9]{2}-[0-9]{2}-[0-9]{2})"
RECORDING_RAW_FULL_REGEX = "_".join([RECORDING_START_REGEX,
                                     RECORDING_COND_REGEX,
                                     RECORDING_RUN_REGEX,
                                     RECORDING_END_REGEX])
RECORDING_SPLIT_FULL_REGEX = "_".join([RECORDING_START_REGEX,
                                       RECORDING_COND_REGEX,
                                       RECORDING_RUN_REGEX,
                                       r"([^_]+)",
                                       RECORDING_END_REGEX])
RECORDING_SPLIT_HIRES_FULL_REGEX = "_".join([RECORDING_START_REGEX,
                                             RECORDING_COND_REGEX,
                                             RECORDING_RUN_REGEX,
                                             r"([^_]+)",
                                             "fullres",
                                             RECORDING_END_REGEX])
