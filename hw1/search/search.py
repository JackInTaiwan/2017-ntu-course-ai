# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    "*** YOUR CODE HERE ***"
    def graphSearch(nodes, nodesRecord) :
        while len(nodes)>0 :
            # Grap the node tended to be expanded
            nodeNow = nodes[-1]
            nodesRecord.append(nodes.pop())

            # Expand the children nodes if the node is not the solution
            if not problem.isGoalState(nodeNow[0]) :
                for successor in problem.getSuccessors(nodeNow[0]) :
                    if True not in [(successor[0] == node[0]) for node in nodesRecord] :
                        pathNew = nodeNow[1][:]  # using [:] to copy
                        pathNew.append(successor[1])
                        nodeNew = (successor[0], pathNew)
                        nodes.append(nodeNew)

            # Reach the solution
            else :
                #print 'Success!'
                return nodeNow[1]

    # Initial the settings
    nodes = []
    nodesRecord = []
    startState = problem.getStartState()

    if not problem.isGoalState(startState) :
        for node in problem.getSuccessors(startState) :
            nodes.append((node[0], [node[1]]))
        return graphSearch(nodes, nodesRecord)
    else :
        return []
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    def graphSearch(nodes, nodesRecord) :
        while len(nodes)>0 :
            # Grap the node tended to be expanded
            nodeNow = nodes[-1]
            nodesRecord.append(nodes.pop())
            print (len(nodes))
            # Expand the children nodes if the node is not the solution
            if not problem.isGoalState(nodeNow[0]) :
                for successor in problem.getSuccessors(nodeNow[0]) :
                    if True not in [(successor[0] == node[0]) for node in nodesRecord] :
                        pathNew = nodeNow[1][:]     # using [:] to copy
                        pathNew.append(successor[1])
                        nodeNew = (successor[0], pathNew)
                        nodes.insert(0, nodeNew)

            # Reach the solution
            else :
                #print 'Success!'
                return nodeNow[1]

    # Initial the settings
    nodes = []
    nodesRecord = []
    startState = problem.getStartState()

    if not problem.isGoalState(startState) :
        for node in problem.getSuccessors(startState) :
            nodes.append((node[0], [node[1]]))
        return graphSearch(nodes, nodesRecord)
    else :
        return []
    #util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def findMinNode (nodes) :
        """
        return: the index in nodes which leads to min-costing node
        """
        minCost = min([node[0] for node in nodes])
        for i, node in enumerate(nodes) :
            if node[0] == minCost :
                return i
    def graphSearch(nodes, nodesRecord) :
        prior = 0
        goal = None
        while len(nodes)>0 :
            # Grap the node tended to be expanded
            nodeNow = nodes[findMinNode(nodes)]
            nodesRecord.append(nodeNow)
            del nodes[findMinNode(nodes)]

            # Expand the children nodes if the node is not the solution
            if not problem.isGoalState(nodeNow[1]) :
                for successor in problem.getSuccessors(nodeNow[1]) :
                    if True not in [(successor[0] == node[1]) for node in nodesRecord] :
                        pathNew = nodeNow[2][:]     # using [:] to copy
                        pathNew.append(successor[1])
                        costNew = nodeNow[0]
                        costNew += successor[2]
                        if goal == None or goal[0] > costNew :
                            nodeNew = [costNew, successor[0], pathNew]
                            nodes.insert(0, nodeNew)
                    else :
                        index = [(successor[0] == node[1]) for node in nodesRecord].index(True)
                        if nodesRecord[index][0] > nodeNow[0] + successor[2] :
                            del nodesRecord[index]
                            pathNew = nodeNow[2][:]  # using [:] to copy
                            pathNew.append(successor[1])
                            costNew = nodeNow[0]
                            costNew += successor[2]
                            nodeNew = [costNew, successor[0], pathNew]
                            nodes.insert(0, nodeNew)

            # Reach the solution
            else :
                goal = nodeNow
        return goal[2]

    nodes = []  # element(one node) = [total cost, state, total path]
    nodesRecord = []
    startState = problem.getStartState()
    if not problem.isGoalState(startState) :
        for node in problem.getSuccessors(startState) :
            nodes.append([node[2], node[0], [node[1]]])
        return graphSearch(nodes, nodesRecord)
    else :
        return []

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def findMinNode (nodes) :
        """
        return: the index in nodes which leads to min-costing node
        """
        minCombinedCost = min([node[0]+node[1] for node in nodes])
        for i, node in enumerate(nodes) :
            if node[0]+node[1] == minCombinedCost :
                return i

    def graphSearch(nodes, nodesRecord) :
        goal = None
        while len(nodes)>0 :
            # Grap the node tended to be expanded
            nodeNow = nodes[findMinNode(nodes)]

            nodesRecord.append(nodeNow)
            del nodes[findMinNode(nodes)]
            # Expand the children nodes if the node is not the solution
            if not problem.isGoalState(nodeNow[2]) :
                for successor in problem.getSuccessors(nodeNow[2]) :
                    if True not in [(successor[0] == node[2]) for node in nodesRecord] :
                        pathNew = nodeNow[3][:]     # using [:] to copy
                        pathNew.append(successor[1])
                        costNew = nodeNow[0]
                        costNew += successor[2]
                        distNew = heuristic(successor[0], problem)
                        if goal == None or goal[0] > costNew :
                            nodeNew = [costNew, distNew, successor[0], pathNew]
                            nodes.insert(0, nodeNew)
                    else :
                        index = [(successor[0] == node[2]) for node in nodesRecord].index(True)
                        if nodesRecord[index][0] > nodeNow[0] + successor[2] :
                            del nodesRecord[index]
                            pathNew = nodeNow[3][:]  # using [:] to copy
                            pathNew.append(successor[1])
                            costNew = nodeNow[0]
                            costNew += successor[2]
                            distNew = heuristic(successor[0], problem)
                            nodeNew = [costNew, distNew, successor[0], pathNew]
                            nodes.insert(0, nodeNew)

            # Reach the solution
            else :
                goal = nodeNow
                return goal[3]
        return []

    nodes = []  # element(one node) = [total cost, dist, state, total path]
    nodesRecord = []
    startState = problem.getStartState()
    if not problem.isGoalState(startState) :
        nodesRecord.append([0, heuristic(startState, problem), startState, []])
        for node in problem.getSuccessors(startState) :
            nodes.append([node[2], heuristic(node[0], problem), node[0], [node[1]]])
        return graphSearch(nodes, nodesRecord)
    else :
        return []
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
