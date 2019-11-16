#! /usr/bin/env python3

# import time
import cv2

Alpha_image   = "/home/jorge/share/Alpha/Pictures/capture.jpg"
Beta_image    = "/home/jorge/share/Beta/Pictures/capture.jpg"
Charlie_image = "/home/jorge/share/Charlie/Pictures/capture.jpg"
Delta_image   = "/home/jorge/share/Delta/Pictures/capture.jpg"

def main():
    """
    Create windows for the images
    """
    cv2.namedWindow('Alpha')
    cv2.moveWindow( 'Alpha', 0, 0)

    cv2.namedWindow('Beta')
    cv2.moveWindow( 'Beta', 650, 0)

    cv2.namedWindow('Charlie')
    cv2.moveWindow( 'Charlie', 0, 570)

    cv2.namedWindow('Delta')
    cv2.moveWindow( 'Delta', 650, 570)

    while True:
        A_image = cv2.imread(Alpha_image)
        cv2.imshow('Alpha', A_image)

        B_image = cv2.imread(Beta_image)
        cv2.imshow('Beta', B_image)

        C_image = cv2.imread(Charlie_image)
        cv2.imshow('Charlie', C_image)

        D_image = cv2.imread(Delta_image)
        cv2.imshow('Delta', D_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
