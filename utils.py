def logging(file, msg):
    with open(file, "a") as f:
        f.write(str(msg) + "\n")