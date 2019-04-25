# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food_left = sum(int(j) for i in newFood for j in i)

        if food_left > 0:
            food_distances = [manhattanDistance(newPos, (x, y))
                              for x, row in enumerate(newFood)
                              for y, food in enumerate(row)
                              if food]
            shortest_food = min(food_distances)
        else:
            shortest_food = 0
        #print("food left: ", food_left)
        #print("food distances: ", shortest_food)
        #

        if newGhostStates:
            ghost_distances = [manhattanDistance(ghost.getPosition(), newPos)
                               for ghost in newGhostStates]
            shortest_ghost = min(ghost_distances)
            

            if shortest_ghost == 0:
                shortest_ghost = -1000
            else:
                shortest_ghost = -8 / shortest_ghost
             
        else:
            shortest_ghost = 0

        #print("food left: ", food_left)
        #print("food distances: ", shortest_food)
        #print("shortest ghost: ", shortest_ghost) 
        
        return -2 * shortest_food + shortest_ghost - 40 * food_left

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def search_depth(state, depth, agent): 
            if agent == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return search_depth(state, depth + 1, 0)
            else:
                actions = state.getLegalActions(agent)

                if len(actions) == 0:
                    return self.evaluationFunction(state)
                next_states = (
                    search_depth(state.generateSuccessor(agent, action),
                    depth, agent + 1)
                    for action in actions
                    )
                return (max if agent == 0 else min)(next_states)
        
        return max(
            gameState.getLegalActions(0),
            key = lambda x: search_depth(gameState.generateSuccessor(0, x), 1, 1)
            )


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        def min_val(state, depth, agent, alpha, beta):
                if agent == state.getNumAgents():
                    return max_val(state, depth + 1, 0, alpha, beta)

                val = None
                for action in state.getLegalActions(agent):
                    successor = min_val(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                    val = successor if val is None else min(val, successor)

                    if alpha is not None and val < alpha:
                        return val

                    beta = val if beta is None else min(beta, val)

                if val is None:
                    return self.evaluationFunction(state)

                return val

        def max_val(state, depth, agent, alpha, beta):
            assert agent == 0

            if depth > self.depth:
                return self.evaluationFunction(state)

            val = None
            for action in state.getLegalActions(agent):
                successor = min_val(state.generateSuccessor(agent, action), depth, agent + 1, alpha, beta)
                val = max(val, successor)

                if beta is not None and val > beta:
                    return val

                alpha = max(alpha, val)

            if val is None:
                return self.evaluationFunction(state)

            return val

        val, alpha, beta, best = None, None, None, None
        for action in gameState.getLegalActions(0):
            val = max(val, min_val(gameState.generateSuccessor(0, action), 1, 1, alpha, beta))

            if alpha is None:
                alpha, best = val, action
            else:
                alpha, best = max(val, alpha), action if val > alpha else best

        return best

def average(lst):
    lst = list(lst)
    return sum(lst) / len(lst)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expHelper(gameState, deepness, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                deepness += 1
            if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxFinder(gameState, deepness, agent)
            else:
                return expFinder(gameState, deepness, agent)
        
        def maxFinder(gameState, deepness, agent):
            output = [0, -float("inf")]
            pacActions = gameState.getLegalActions(agent)
            
            if not pacActions:
                return self.evaluationFunction(gameState)
                
            for action in pacActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expHelper(currState, deepness, agent+1)
                if type(currValue) is list:
                    testVal = currValue[1]
                else:
                    testVal = currValue
                if testVal > output[1]:
                    output = [action, testVal]                    
            return output
            
        def expFinder(gameState, deepness, agent):
            output = [0, 0]
            ghostActions = gameState.getLegalActions(agent)
            
            if not ghostActions:
                return self.evaluationFunction(gameState)
                
            probability = 1.0/len(ghostActions)    
                
            for action in ghostActions:
                currState = gameState.generateSuccessor(agent, action)
                currValue = expHelper(currState, deepness, agent+1)
                if type(currValue) is list:
                    val = currValue[1]
                else:
                    val = currValue
                output[0] = action
                output[1] += val * probability
            return output
             
        outputList = expHelper(gameState, 0, 0)
        return outputList[0] 

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Pacman goes for pellets over everything, but we give pacman prices for going after ghosts as well.
      When the ghost is close to pacman we give pacman less points.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    if currentGameState.isWin():  return float("inf")
    if currentGameState.isLose(): return float("-inf")
    
    # Fetch several data we require to analyze thecurrent state of the pacman's environment
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates() 
    foodPos = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    def manhattan(xy1, xy2):
      return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    manhattans_food = [ ( manhattan(pacmanPos, food) ) for food in foodPos.asList() ]
    min_manhattans_food = min(manhattans_food)

    manhattans_ghost = [ manhattan(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates if ghostState.scaredTimer == 0 ]
    min_manhattans_ghost = -3
    
    if ( len(manhattans_ghost) > 0 ): 
      min_manhattans_ghost = min(manhattans_ghost)

    manhattans_ghost_scared = [ manhattan(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates if ghostState.scaredTimer > 0 ]
    min_manhattans_ghost_scared = 0;
    
    if ( len(manhattans_ghost_scared) > 0 ): 
      min_manhattans_ghost_scared = min(manhattans_ghost_scared)

    
    score = scoreEvaluationFunction(currentGameState)
    score += -1.5 * min_manhattans_food
    score += -2 * ( 1.0 / min_manhattans_ghost )
    score += -2 * min_manhattans_ghost_scared
    score += -20 * len(capsules)
    score += -4 * len(foodPos.asList())

    return score

# Abbreviation
better = betterEvaluationFunction

