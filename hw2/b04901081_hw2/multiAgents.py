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
        chosenIndex = random.choice(bestIndices) if len(bestIndices) > 1 else bestIndices[0] # Pick randomly among the best
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
        currentFood = list(currentGameState.getFood())
        currentFoodPos = []
        newFood = list(successorGameState.getFood())
        newFoodPos = []
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # ghostState.configuration.pos is a float
        # newGhostPoses = [(1.0,4.0), (2.0,13.0)]
        currentGhostPoses = [ghostState.configuration.pos for ghostState in newGhostStates]
        currentCapsules = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        def euDistance(x, y) :
            return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

        def manDistance(x, y) :
            return abs(x[0]-y[0]) + abs(x[1]-y[1])

        ghosts = []
        ghostsDises = []
        scaredGhosts = []

        for i, time in enumerate(newScaredTimes):
            if time == 0:
                ghosts.append(i)
            else:
                scaredGhosts.append(i)
        ###Find nearest one
        for i in ghosts:
            dis = euDistance(currentGhostPoses[i], newPos)
            ghostsDises.append(dis)

        ### Create currentFoodPos
        def createCurrentFoodPos() :
            for i in range(len(currentFood)) :
                for j in range(len(currentFood[0])) :
                    if currentFood[i][j] == True : currentFoodPos.append((i,j))

        ### Create newFoodPos
        def createNewFoodPos() :
            for i in range(len(newFood)) :
                for j in range(len(newFood[0])) :
                    if newFood[i][j] == True : newFoodPos.append((i,j))

        ### Use the nearest food to evaluate score
        def findNearestFoodDisScore(newPos, newFoodPos, currentFoodPos) :
            if newPos in currentFoodPos :
                return -100
            else :
                foodDises = []
                for food in newFoodPos :
                    dis = euDistance(newPos, food)
                    foodDises.append(dis)
                foodDisesSelected = sorted(foodDises)[0]
                # coeff. and noise(to avoid lingering)
                k = 100 if len(scaredGhosts) == 0 else 1
                noise = 1 + random.random()
                return foodDisesSelected * noise * k

        ### use ghost to evaluate score
        def ghostScore(newPos, currentGhostPoses, newGhostScaredTimes, action) :
            ghosts = []
            ghostsDises = []
            scaredGhosts = []
            for i, time in enumerate(newGhostScaredTimes) :
                if time == 0 : ghosts.append(i)
                else : scaredGhosts.append(i)
            ###Find nearest one
            for i in ghosts :
                dis = manDistance(currentGhostPoses[i], newPos)
                ghostsDises.append(dis)
            nearestGhostDis = min(ghostsDises) if len(ghostsDises) > 0 else 999999

            if action == 'Stop' : return -999999999
            else :
                if len(ghosts) > 0 :
                    nearestLimit = 3 + len(ghosts)
                    if nearestGhostDis < nearestLimit :
                        return -999999 * (nearestLimit-nearestGhostDis)
                    elif len(scaredGhosts) > 0 :
                        scaredGhostScores = [newGhostScaredTimes[i]/euDistance(newPos, currentGhostPoses[i]) for i in scaredGhosts]

                        return 16000 * (len(ghosts)) + 100 * max(scaredGhostScores)
                    else :
                        return 16000 + 160 * 100 + nearestGhostDis + 100
                else :
                    scaredGhostScores = [newGhostScaredTimes[i] / euDistance(newPos, currentGhostPoses[i]) for i in
                                         scaredGhosts]
                    return 16000 * (len(ghosts)) + 100 * max(scaredGhostScores)


        def capsuleScore(newPos, currentCapsules) :
            if newPos in currentCapsules : return -float('inf')
            elif len(currentCapsules) > 0 :
                k = 200 if len(scaredGhosts) == 0 else 1
                dis = min([euDistance(capsule, newPos) for capsule in currentCapsules])
                return dis * k
            else : return 0

        createNewFoodPos()
        createCurrentFoodPos()
        nearestFoodDis = findNearestFoodDisScore(newPos, newFoodPos, currentFoodPos)
        score = ghostScore(newPos, currentGhostPoses, newScaredTimes, action) - nearestFoodDis - capsuleScore(newPos, currentCapsules)
        return score

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

        def recursiveFindMaxMin(self, gameState, count):
            index = count % gameState.getNumAgents()
            if count == 0:
                count += 1
                vals = [(recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count), action) for action in gameState.getLegalActions(index)]
                return max(vals)[1]
            elif count > self.depth * gameState.getNumAgents() - 1 or gameState.isWin() or gameState.isLose() :
                return self.evaluationFunction(gameState)
            else :
                count += 1
                vals = [recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count) for action in gameState.getLegalActions(index)]
                return max(vals) if index ==0 else min(vals)

        return recursiveFindMaxMin(self, gameState, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def recursiveFindMaxMin(self, gameState, count, alBe):
            index = count % gameState.getNumAgents()
            alBe = alBe[:]
            if count == 0:
                count += 1
                vals = []
                for action in gameState.getLegalActions(index) :
                    score = recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count, alBe)
                    if alBe[1] >= score >= alBe[0] :
                        vals.append((score, action))
                        if index == 0 :
                            alBe[0] = score
                        else :
                            alBe[1] = score
                    elif (index == 0 and score > alBe[1]) or (index != 0 and score < alBe[0]) : break
                final = [valPair[1] for valPair in vals if valPair[0] == max(vals)[0]]
                return random.choice(final)
            elif count > self.depth * gameState.getNumAgents() - 1 or gameState.isWin() or gameState.isLose() :
                return self.evaluationFunction(gameState)
            else :
                count += 1
                vals = []
                for action in gameState.getLegalActions(index) :
                    score = recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count, alBe)
                    if alBe[1] >= score >= alBe[0] :
                        vals.append(score)
                        if index == 0 :
                            alBe[0] = score
                        else :
                            alBe[1] = score
                    elif (index == 0 and score > alBe[1]) or (index != 0 and score < alBe[0]) :
                        return None
                if len(vals) > 0 : return max(vals) if index == 0 else min(vals)
                else : return None


        return recursiveFindMaxMin(self, gameState, 0, [-float('inf'), float('inf')])

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
        def recursiveFindMaxMin(self, gameState, count):
            index = count % gameState.getNumAgents()
            if count == 0:
                count += 1
                vals = [(recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count), action) for action in gameState.getLegalActions(index)]
                return max(vals)[1]
            elif count > self.depth * gameState.getNumAgents() - 1 or gameState.isWin() or gameState.isLose() :
                return self.evaluationFunction(gameState)
            else :
                count += 1
                vals = [recursiveFindMaxMin(self, gameState.generateSuccessor(index, action), count) for action in gameState.getLegalActions(index)]
                return max(vals) if index ==0 else sum(vals)/len(vals)

        return recursiveFindMaxMin(self, gameState, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      ### Functions
      
        initialGhostsScaredghosts: 
            Find and separate the indexes of ghosts and scaredGhosts, put them into
            to lists individually.
        
        euDistance:
            Calculate the distance by Euclidean distance.
        
        manDistance:
            Calculate the distance by Manhattan distance.
            
        createCurrentFoodPos:
            Transfer the type of food into list format.
        
        findNearestFoodDisScore: 
            Score by the nearestFood.
            Plus, we calculate weighted scored depend on whether there exit 
            scaredGhosts. If there's no scaredGhosts, the weight is high, by the 
            contrary, we make the weight low for we desire to chasing scaredGhosts
            more for significant scores instead of the relatively obscure scored from
            food.
            What's more, in order to avoid lingering around due to the symmetric attribution
            of food, we add a noisy term to solve it.
            
        capsuleScore:
            Score by the nearestCapsule.
            The nearer capsule, the higher weighted score.
            Note the this function return 0 (no effect) when the capsules are all eaten.
            
        ghostScore:
            Score by the distances, attribution and amount of ghosts and scaredGhosts.
            If there is no scaredGhosts, the farther distance from the nearest ghost brings
            out higher score, but it matters not so much, so the total score is now domonated 
            by food score and capsule food in this case; however, when the nearset ghost is too
            close, the score tremendous decreases to escape as the nearest ghost comes closer.
            Even in case that no scaredGhost and no ghost is too close, we still need to be careful
            when the total distance from pacman to the two ghosts because they may surround pacman
            which probably leads to kill pacman with no any chance to escape. So, we take the total
            distance from pacman to the two ghosts into consideration.
            When there is scaredGhost, we design a significant score for chasing it.
         
      ### Main Loop
        three cases below :
            if it's a lose state, return dominantly negative score dependent on the final score.
            if it's a win state, return dominantly positive score dependent on the final score.
            else, return the total score calculated by the functions mentioned above.
    """
    "*** YOUR CODE HERE ***"
    ### Initialize state that we need
    currentPos = currentGameState.getPacmanPosition()
    currentFood = list(currentGameState.getFood())
    currentFoodPos = []
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # ghostState.configuration.pos is a float
    # newGhostPoses = [(1.0,4.0), (2.0,13.0)]
    currentGhostPoses = [ghostState.configuration.pos for ghostState in ghostStates]
    currentCapsules = currentGameState.getCapsules()
    ghosts = []
    scaredGhosts = []



    ### Get ghosts and scaredGhosts index
    def initialGhostsScaredghosts(ghosts, scaredGhosts) :
        for i, time in enumerate(scaredTimes):
            if time == 0:
                ghosts.append(i)
            else:
                scaredGhosts.append(i)

    def euDistance(x, y):
        return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5

    def manDistance(x, y):
        return abs(x[0] - y[0]) + abs(x[1] - y[1])

    ### Create currentFoodPos
    def createCurrentFoodPos():
        for i in range(len(currentFood)):
            for j in range(len(currentFood[0])):
                if currentFood[i][j] == True: currentFoodPos.append((i, j))

    ### Use the nearest food to evaluate score
    def findNearestFoodDisScore(newPos, currentFoodPos):
            foodDises = []
            for food in currentFoodPos:
                dis = euDistance(newPos, food)
                foodDises.append(dis)
            foodDisesSelected = sorted(foodDises)[0]
            # coeff. and noise(to avoid lingering)
            k = 50 if len(scaredGhosts) == 0 else 1
            noise = 1 + random.random() * 1.5
            return len(currentFoodPos) * k * 50 + foodDisesSelected * noise * k

    ### CapsuleScore
    def capsuleScore(newPos, currentCapsules):
        k = 100 if len(scaredGhosts) == 0 else 1
        if len(currentCapsules) > 0 :
            dis = min([euDistance(capsule, newPos) for capsule in currentCapsules])
            return dis * k
        else : return 0

    ### use ghost to evaluate score
    def ghostScore(newPos, currentGhostPoses, scaredTimes, ghosts, scaredGhosts):
        ### Cases for ghosts and scaredGhosts
        ghostsDises = []
        eatCapsuleScore = 10000000
        threatGhostsTotalDis = 10
        if len(ghosts) > 0:
            eatScaredGhostScore = 160000
            # Find ghost dises and the nearest one
            for i in ghosts:
                dis = manDistance(currentGhostPoses[i], newPos)
                ghostsDises.append(dis)
            ghostsDises = sorted(ghostsDises)
            nearestGhostDis = ghostsDises[0]

            # score for the distance from ghosts and scaredGhosts
            disLimit = 2 + len(ghosts)
            if nearestGhostDis < disLimit:
                totalThreatDis = 0
                chasedThreat = -999999999
                for dis in ghostsDises:
                    if dis < disLimit:
                        totalThreatDis += (disLimit - dis)
                return chasedThreat * totalThreatDis
            elif len(ghostsDises) > 1 and sum(ghostsDises) < threatGhostsTotalDis and manDistance(currentGhostPoses[0], currentGhostPoses[1]) > 0 :
                    chasedThreat = -9999999
                    return chasedThreat * (threatGhostsTotalDis - sum(ghostsDises))
            elif len(scaredGhosts) > 0:
                scaredGhostScores = [scaredTimes[i] / euDistance(newPos, currentGhostPoses[i]) for i in
                                         scaredGhosts]
                return eatCapsuleScore * (2-len(currentCapsules)) + eatScaredGhostScore * len(ghosts) + 100 * max(scaredGhostScores)
            else:
                return eatCapsuleScore * (2-len(currentCapsules)) + eatScaredGhostScore * len(ghosts) + 100 * 160 + nearestGhostDis * 10
        else:
            scaredGhostScores = [scaredTimes[i] / euDistance(newPos, currentGhostPoses[i]) for i in
                                     scaredGhosts]
            return eatCapsuleScore * (2-len(currentCapsules)) + 16000 * 0 + 100 * max(scaredGhostScores)



    if currentGameState.isWin() :
        winBasicScore = 999999999999
        return winBasicScore + currentGameState.getScore()
    elif currentGameState.isLose() :
        loseBasicScore = -999999999999
        return loseBasicScore + currentGameState.getScore()
    else :
        initialGhostsScaredghosts(ghosts, scaredGhosts)
        createCurrentFoodPos()
        nearestFoodDis = findNearestFoodDisScore(currentPos, currentFoodPos)
        score = ghostScore(currentPos, currentGhostPoses, scaredTimes, ghosts, scaredGhosts) - nearestFoodDis - capsuleScore(currentPos, currentCapsules)
        return score


# Abbreviation
better = betterEvaluationFunction