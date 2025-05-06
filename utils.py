def logging(file, msg):
    with open(file, "a") as f:
        f.write(msg + "\n")