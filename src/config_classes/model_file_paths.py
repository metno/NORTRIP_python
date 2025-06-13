from dataclasses import dataclass


@dataclass
class model_file_paths:
    """
    Dataclass containing file paths and names
    """

    path_inputparam = ""
    path_inputdata = ""
    path_outputdata = ""
    path_outputfig = ""
    path_ospm = ""
    path_fortran = ""
    path_fortran_output = ""
    filename_inputparam = ""
    filename_inputdata = ""
    filename_outputdata = ""
    filename_outputfigures = ""
    filename_log = ""
    file_fortran_exe = ""
    title_str = ""
    path_filename_inputparam = ""
    path_filename_inputdata = ""
