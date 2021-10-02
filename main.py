

from fasttext import train_supervised

def print_results(N, p, r):
    print("Number of samples\t" + str(N))
    print("Precision\t{:.3f}".format(p))
    print("Recall\t{:.3f}".format(r))


if __name__ == "__main__":
    train_data = "C:\\Users\\nourm\\PycharmProjects\\spamIdentification\\traindata.txt"
    valid_data = "C:\\Users\\nourm\\PycharmProjects\\spamIdentification\\testdata.txt"

    # train_supervised uses the same arguments and defaults as the fastText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
    print_results(*model.test(valid_data))
    model.save_model("spam.bin")
