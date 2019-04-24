node = state
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)
    
    unvisitedCorners = []
    sum = 0
    for corner in corners:
        if not corner in visitedCorners_:
            unvisitedCorners.append(corner)

    currentPoint = node
    while len(unvisitedCorners) > 0:
        distance, corner = min([(util.manhattanDistance(currentPoint, corner), corner) for corner in unvisitedCorners])
        sum += distance
        currentPoint = corner
        unvisitedCorners.remove(corner)

    print "Heuristic: ", sum
return sum
