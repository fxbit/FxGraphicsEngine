
#include "includes.h"

#ifndef H_BOUNDARY_UTILS
#define H_BOUNDARY_UTILS

// ------------ InitNewBoundaryNode ------------ //

// init a new HalfEdge with know startVertex and return the id
__device__
uint InitNewBoundaryNode(const DATA_TYPE    *VertexList, 
                         const HalfEdge     *HEList,
                         ThreadInfo         *threadInfo,
                         uint                halfEdgeID, 
                         BoundaryNode       *newBN)
{
    // get and inc the last id for halfEdge
    uint newBN_ID = threadInfo->lastBoundaryNodeID;
    threadInfo->lastBoundaryNodeID ++;

    // fill the fields
    newBN->halfEdgeID = halfEdgeID;
 
    // link the node with infinity...
    newBN->PrevNodeID = UNSET;
    newBN->NextNodeID = UNSET;
    
    // return the new newBN_ID
    return newBN_ID;
}



// ------------ AddOutsideVertex ------------ //

__device__
void AddOutsideVertex( const DATA_TYPE  *VertexList, 
                       HalfEdge			*HEList,
                       BoundaryNode     *BoundaryList,
                       Face             *FaceList,
                       DelaunayNode     *stack,
                       uint              vertID,
                       ThreadInfo       *threadInfo)
{
    // get the specific vertex 
    DATA_TYPE vert = VertexList[vertID];

    // start the node from the start
    uint 			tmpNodeID 	= threadInfo->boundaryNodeRootID;
    BoundaryNode 	tmpNode 	= BoundaryList[tmpNodeID];
    
    // help variables
    uint 			firstNodeID = UNSET;
    uint		 	lastNodeID  = UNSET;
    BoundaryNode	firstNode, lastNode;
    
    bool			foundTheFirst 		= false; 
    bool			firstMatchIsRoot 	= false;
    bool			TooClose 			= false;
    bool 			AcceptSide 			= false;
    
    // =============================================================================== //
    // Forward Test
    // pass all the nodes and find the first and the last one that we can use
    do
    {
        // load the current edge in the boundary list
        HalfEdge he = HEList[tmpNode.halfEdgeID];
        
        // load the vertex of the half edge
        DATA_TYPE heStartVertex = VertexList[he.startVertexID];
        
        // load the current edge in the boundary list
        HalfEdge heNext = HEList[he.nextEdgeID];
        
        // load the next start vertex
        DATA_TYPE heNextStartVertex = VertexList[heNext.startVertexID];
        
        AcceptSide = SideTest(heStartVertex, heNextStartVertex, vert );
        
        // test the angle if is negative
        // (neg mean that we are outside from the mesh )
        if(AcceptSide)
        {
            // check if we have find allready the first accepted node
            if(foundTheFirst){
                lastNodeID = tmpNodeID;
            }else{
                
                // set the first node
                firstNodeID = tmpNodeID;
                
                // set the last node
                lastNodeID = tmpNodeID;
                
                // set the flag that we have find the first 
                foundTheFirst = true;
            }
            
        } else {
            if (foundTheFirst)
                break;
        }
        
        // go to the next node
        tmpNodeID = tmpNode.NextNodeID;
        
        // load the next node
        tmpNode = BoundaryList[tmpNodeID];
    } while (tmpNodeID != threadInfo->boundaryNodeRootID);

    // =============================================================================== //
    
    if (!foundTheFirst)
        return;
    
    // pass all the nodes and find the first and the last one that we can use
    if(firstMatchIsRoot){
        // reset the variables to go the prev node from the root
        tmpNodeID 	= threadInfo->boundaryNodeRootID;
        tmpNode 	= BoundaryList[tmpNodeID];
        tmpNodeID 	= tmpNode.PrevNodeID;
        tmpNode 	= BoundaryList[tmpNodeID];
        
        while (true)
        {
            // load the current edge in the boundary list
            HalfEdge he = HEList[tmpNode.halfEdgeID];
            
            // load the vertex of the half edge
            DATA_TYPE heStartVertex = VertexList[he.startVertexID];
            
            // load the current edge in the boundary list
            HalfEdge heNext = HEList[he.nextEdgeID];
            DATA_TYPE heNextStartVertex = VertexList[heNext.startVertexID];
            
            // test the angle if is negative
            // (neg mean that we are outside from the mesh )
            if( SideTest(heStartVertex, heNextStartVertex, vert ) )
            {
                // set the first node 
                firstNodeID = tmpNodeID;
            }
            else
                break;
            
            // go to the prev node
            tmpNodeID = tmpNode.PrevNodeID;
            
            // check that we are in the beginning 
            if(tmpNodeID == threadInfo->boundaryNodeRootID)
                break;
            
            // load the prev node
            tmpNode = BoundaryList[tmpNodeID];
        }
        
    }

    
    // =============================================================================== //
    // Link the new nodes
    
    // pass the nodes from the first to last 
    tmpNodeID 	= firstNodeID;
    tmpNode 	= BoundaryList[tmpNodeID];
    
    foundTheFirst 	= false;
    
    BoundaryNode prevEndNode = BoundaryList[lastNodeID];
    uint endNodeID = prevEndNode.NextNodeID;
    
    while( tmpNodeID != endNodeID)
    {
        // load the current edge in the boundary list
        uint heID       = tmpNode.halfEdgeID;
        HalfEdge he 	= HEList[heID];
        HalfEdge heNext = HEList[he.nextEdgeID];
        
        
        // create 3 new he
        HalfEdge he1, he2, he3;
        uint 	 he1ID, he2ID, he3ID;
        bool	 ccw;

        // create the first triangle
        CreateTriangle<false>(VertexList, HEList, threadInfo, 
                             he.startVertexID, vertID, heNext.startVertexID,
                             &he1, &he2, &he3,
                             &he1ID, &he2ID, &he3ID,
                             &ccw);
        
        // link the twin edge
        he.twinEdgeID 	= he3ID;
        he3.twinEdgeID 	= heID;
        
        // if we are in the mid of the search 
        // link the prev edge to the triangle
        if (foundTheFirst)
        {
            BoundaryNode prevNode = BoundaryList[tmpNode.PrevNodeID];
            HalfEdge prevHe = HEList[prevNode.halfEdgeID];
            
            // link the twin edges
            he1.twinEdgeID = prevNode.halfEdgeID;
            prevHe.twinEdgeID = he1ID;
            
            // store the prev he
            HEList[prevNode.halfEdgeID] = prevHe;
            
            // check if we are going to remove the root node
            if(  tmpNode.PrevNodeID == threadInfo->boundaryNodeRootID)
            {
                // change the root node	
                threadInfo->boundaryNodeRootID  = prevNode.PrevNodeID;
            }
            
            // remove the prev node
            {
                // get prev/next nodes
                BoundaryNode prevPrevNode = BoundaryList[prevNode.PrevNodeID];
                
                // change the prev/next pointers
                prevPrevNode.NextNodeID = tmpNodeID;
                tmpNode.PrevNodeID = prevNode.PrevNodeID;
                
                // store the changes
                BoundaryList[prevNode.PrevNodeID] = prevPrevNode;
            }
            
            // replace the half edge id of the tmpNode
            tmpNode.halfEdgeID = he2ID;
            
            // store the changed bn
            BoundaryList[tmpNodeID]=tmpNode;
        }else{ /* !foundTheFirst */
            
            // replace the boundary node with the 2 news by using the old one and add one new
            tmpNode.halfEdgeID = he1ID;
            
            // add a new BN after the current one
            {
                BoundaryNode newNode;
                uint newNodeID;
                
                // get the next node 
                uint nextNodeID = tmpNode.NextNodeID;
                BoundaryNode nextNode = BoundaryList[nextNodeID];
                
                // create a new node 
                newNodeID = InitNewBoundaryNode(VertexList, HEList, threadInfo,
                                                he2ID, &newNode);
                
                // set the next node to point to the new node
                nextNode.PrevNodeID = newNodeID;
                
                // set the new node to point to the next node
                newNode.NextNodeID = tmpNode.NextNodeID;
                
                // set this node to point to the new node
                tmpNode.NextNodeID = newNodeID;
                
                // link the new node to point to this node
                newNode.PrevNodeID = tmpNodeID;
                
                
                //// store all the changes
                BoundaryList[nextNodeID] = nextNode;
                BoundaryList[newNodeID]  = newNode;
                BoundaryList[tmpNodeID]  = tmpNode;
                
                // move to the next one 
                tmpNodeID = newNodeID;
                tmpNode= newNode;
            }
            
            // set that we have find the first that is correct
            foundTheFirst = true;
        }
        
        // store the he
        HEList[heID] = he;
        HEList[he1ID] = he1;
        HEList[he2ID] = he2;
        HEList[he3ID] = he3;
        
        // create a new face
        CreateFace(HEList,
                   FaceList,
                   he1ID,  
                   threadInfo);
        
        // TODO: delaunay correction
        // add to stack all the corrections that want to have
        {
            // create face for fixes
            DelaunayNode newNode;
            newNode.RootHalfEdgeID 	= he1.nextEdgeID;
            HalfEdge h = HEList[he1.nextEdgeID];
            h = HEList[h.nextEdgeID];
            newNode.NextFaceHalfEdgeID = h.twinEdgeID;
            
            // push the face that we want to fix
            PushDelaunayNode( stack, 
                              newNode, 
                              threadInfo );
        }
        // move to the next node
        tmpNodeID = tmpNode.NextNodeID;
        
        // load the next node
        tmpNode = BoundaryList[tmpNodeID];
    }
}










#endif /* H_BOUNDARY_UTILS */





