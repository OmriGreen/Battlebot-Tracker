import numpy as np
import cv2 as cv
import gradio as gr

# calculates the transformation matrix for the birdseye view of the arena given an image of the arena and the width of the arena in feet
class transform_img:
    def __init__(self, setupImg, arena_width, scale):
        self.setupImg = setupImg
        self.squarePts = None
        self.transformMatrix = None
        self.arena_width = arena_width  # width / height of the arena in feet
        self.scale = scale
        self.M = self.setup_transform() #Transformation Matrix

    # Changes to grayscale
    def grayScale(self,img):
        return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    def order_points(self, pts):
        # pts: array of shape (4,2)
        rect = np.zeros((4, 2), dtype="float32")

        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]      # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]      # bottom-right has largest sum
        rect[1] = pts[np.argmin(diff)]   # top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]   # bottom-left has largest difference

        return rect


    # Edge Detection
    def edgeDetection(self,img, thresh1, thresh2):
        # Apply Gaussian Blur to reduce noise
        blur = cv.GaussianBlur(img, (5, 5), 1.4)
        
        # Apply Canny Edge Detector
        edges = cv.Canny(blur, threshold1=thresh1, threshold2=thresh2)
        return edges
    
    # Gets Lines from Hough Transform
    def getLines(self,edges):
        lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                               minLineLength=250, maxLineGap=200)

        if lines is None or len(lines) == 0:
            print("squarePoints: no Hough lines found")
            return None

        # Sort lines by length
        lengths = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            lengths.append((length, (int(x1), int(y1), int(x2), int(y2))))

        # Get the 2 largest lines
        lengths.sort(key=lambda x: x[0], reverse=True)
        
        
        lines = []
        for length in lengths:
            lines.append(length[1])
        
        return lines
    
    # Sets up the transformation matrix for birdseye view
    def setupTransformMatrix(self, squarePts, width, height):
        if squarePts is None or len(squarePts) < 4:
            print("setupTransformMatrix: Not enough points to define a square.")
            return None
        
        self.squarePts = np.array(squarePts, dtype="float32")
        # Define the destination points for the birdseye view
        

        scale = self.scale  # pixels per foot, arbitrary scaling
        dstPts = np.array([
            [0, 0],
            [self.arena_width * scale, 0],
            [self.arena_width * scale, self.arena_width * scale],
            [0, self.arena_width * scale]
        ], dtype="float32")

        # Compute the perspective transformation matrix
        self.transformMatrix = cv.getPerspectiveTransform(self.squarePts, dstPts)

        return self.transformMatrix
    # Merges lines
    def merge_lines(self, lines):
        new_lines = []
        for line1 in lines:
            for line2 in lines:
                if line1 is line2:
                    continue
                x11, y11, x12, y12 = line1
                x21, y21, x22, y22 = line2
                
                # Calculate slopes
                if (x12 - x11) == 0 or (x22 - x21) == 0:
                    continue  # Avoid division by zero for vertical lines
                
                slope1 = (y12 - y11) / (x12 - x11)
                slope2 = (y22 - y21) / (x22 - x21)
                
                # Check if slopes are similar (within a small threshold)
                if abs(slope1 - slope2) < 0.1:
                    # Check if lines are close to each other
                    dist1 = np.hypot(x21 - x11, y21 - y11)
                    dist2 = np.hypot(x22 - x12, y22 - y12)
                    
                    if dist1 < 10 or dist2 < 10:
                        # Merge lines by taking the endpoints that are farthest apart
                        new_line = (min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))
                        new_lines.append(new_line)
                        break  # No need to check other lines once merged
        # Removes duplicates
        new_lines = list(set(new_lines))

        # Removes lines below a certain length
        filtered_lines = []
        for line in new_lines:
            x1, y1, x2, y2 = line
            length = np.hypot(x2 - x1, y2 - y1)
            if length > 500:  # Minimum length threshold
                filtered_lines.append(line)
        return filtered_lines
   
    # Finds intersection point of all lines
    def line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        
        if px < 0 or py < 0:
            return None  # Intersection point is out of bounds (negative coordinates)
        return int(px), int(py)
    
    # Finds all intersection points from lines
    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self.line_intersection(lines[i], lines[j])
                if pt is not None:
                    intersections.append(pt)
        return intersections

    # Finds the two lowest points from the lines
    def find_lowest_point(self, lines):
        lowest_points = []
        for line in lines:
            x1, y1, x2, y2 = line
            if y1 > y2:
                lowest_points.append((x1, y1))
            else:
                lowest_points.append((x2, y2))
        
        # Sort points by their y-coordinate (lowest first)
        lowest_points.sort(key=lambda pt: pt[1], reverse=True)
        
        # Return the two lowest points
        return lowest_points[0]
    
    # Finds all points on the square given the lowest points and top segment
    def find_square(self, lines, lowest_point, top_segment):
        if lowest_point is None or top_segment is None or len(top_segment) < 2:
            return None
        # Choose the two points in top_segment that are farthest apart
        max_d = -1
        p1 = top_segment[0]
        p2 = top_segment[1]
        for i in range(len(top_segment)):
            for j in range(i + 1, len(top_segment)):
                a = top_segment[i]
                b = top_segment[j]
                d = np.hypot(b[0] - a[0], b[1] - a[1])
                if d > max_d:
                    max_d = d
                    p1, p2 = a, b

        # Compute delta from p1 to p2 (ensure correct orientation)
        x1, y1 = p1
        x2, y2 = p2
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Make a fake line using the lowest point and the delta from the top segment
        fake_line = (int(lowest_point[0]), int(lowest_point[1]), int(lowest_point[0] + delta_x), int(lowest_point[1] + delta_y))
        intersections = [fake_line]
        for line in lines:
            if line != top_segment:
                intersections.append(line)
        square = self.find_intersections(intersections)
        square = self.order_points(np.array(square))      
        return square
    
    # Sets up the image automatically based on the given data
    def setup_transform(self):
        # Edge Detection
        edges = self.edgeDetection(self.grayScale(self.setupImg),70,400)
        # Gets all lines from Hough Lines
        lines = self.getLines(edges)
        # merges lines
        merged_lines = self.merge_lines(lines)

        # finds intersection of lines to find the top segment square
        topSegment = self.find_intersections(merged_lines)

        # Finds the 2 lowest points in the lines to set as the bottom of the square
        bottomPoint = self.find_lowest_point(merged_lines)

        # Finds all points on the square given the lowest points and top segment
        square = self.find_square(merged_lines, bottomPoint, topSegment)
        print("square from find_square:", square)

        self.M = self.setupTransformMatrix(square, self.arena_width*self.scale, self.arena_width*self.scale)

    # Transforms the image to a birdseye view using the transformation matrix
    def transform(self, img):
        if self.M is None:
            print("transform: Transformation matrix not set up.")
            return None
        warped = cv.warpPerspective(img, self.M, (self.arena_width*self.scale, self.arena_width*self.scale))
        return warped