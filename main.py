from detectorlib import window


def main():
    window_ = window.Window()
    window_.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print(ex)
