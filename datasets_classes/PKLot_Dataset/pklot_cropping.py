import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

directory = "E:/datasety/pklot/correct/PKLot.tar/PKLot"
subdir = "PKLot"


# TODO - crop spaces from images, decide if this is actually needed, now it's just a test and not used anywhere,
#  non-functional

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = pts
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = ((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5
    widthB = ((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2) ** 0.5
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = ((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2) ** 0.5
    heightB = ((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2) ** 0.5
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def test():
    jpg_file = "E:/datasety/pklot/correct/PKLot.tar/PKLot/PKLot/UFPR04/Sunny/2012-12-11/2012-12-11_17_11_09.jpg"
    xml_file = "E:/datasety/pklot/correct/PKLot.tar/PKLot/PKLot/UFPR04/Sunny/2012-12-11/2012-12-11_17_11_09.xml"

    """
    <parking id="pucpr">
      <space id="1" occupied="0">
        <rotatedRect>
          <center x="300" y="207" />
          <size w="55" h="32" />
          <angle d="-74" />
        </rotatedRect>
        <contour>
          <point x="278" y="230" />
          <point x="290" y="186" />
          <point x="324" y="185" />
          <point x="308" y="230" />
        </contour>
      </space>
      ...
    """

    root = ET.parse(xml_file).getroot()
    all_points = []

    for space in root.findall('space'):
        space_id = space.attrib['id']
        occupied = space.attrib['occupied']
        rotated_rect = space.find('rotatedRect')
        center = rotated_rect.find('center')
        size = rotated_rect.find('size')
        angle = rotated_rect.find('angle')
        contour = space.find('contour')
        points = []
        for point in contour.findall('point'):
            points.append((int(point.attrib['x']), int(point.attrib['y'])))
        all_points.append(points)

        print(f"Space {space_id}:")
        print(f"Occupied: {occupied}")
        print(f"Center: {center.attrib['x']}, {center.attrib['y']}")
        print(f"Size: {size.attrib['w']}, {size.attrib['h']}")
        print(f"Angle: {angle.attrib['d']}")
        print(f"Points: {points}")

    img = cv2.imread(jpg_file)
    idx = 0
    for points in all_points:
        idx += 1
        for i in range(len(points)):
            cv2.line(img, points[i], points[(i + 1) % len(points)], (0, 255, 0), 2)
            image_perspective = four_point_transform(img, points)
            cv2.imshow('image', image_perspective)
            cv2.waitKey(0)
        # put text into the middle of the parking space
        x = sum([p[0] for p in points]) // len(points)
        y = sum([p[1] for p in points]) // len(points)
        cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # main()
    test()
