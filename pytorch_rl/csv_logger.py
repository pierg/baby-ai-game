import csv


current_csv_file = ""

def create_header(csv_file, headers):
    """
    Creates headers in a csv file
    :param csv_file: A csv file to write to
    :param headers: a list containing the headers wanted, e.g ['Date', 'Tempratrue 1', ... ]
    :return: void
    """
    global current_csv_file
    current_csv_file = csv_file
    with open(csv_file, 'w', newline='') as csv_log:
        writer = csv.writer(csv_log, delimiter=',')
        writer.writerow(headers)

def write_to_log(log_text):
    """
    Writes a row to a csv log file, appends to an existing file
    :param csv_file: A csv file to append to
    :param log_text: A row to add to the file, make sure it follows the headers specified, expects a string with comma
    seperated values, e.g "51,73,735"
    :return: void
    """
    with open(current_csv_file, 'a', newline='') as csv_log:
        writer = csv.writer(csv_log, delimiter=',')
        writer.writerow(log_text.split(","))
