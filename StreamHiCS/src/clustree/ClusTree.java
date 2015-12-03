package clustree;

/*
*    ClusTree.java
*    Copyright (C) 2010 RWTH Aachen University, Germany
*    @author Sanchez Villaamil (moa@cs.rwth-aachen.de)
*
  *    This program is free software; you can redistribute it and/or modify
  *    it under the terms of the GNU General Public License as published by
  *    the Free Software Foundation; either version 3 of the License, or
  *    (at your option) any later version.
  *
  *    This program is distributed in the hope that it will be useful,
  *    but WITHOUT ANY WARRANTY; without even the implied warranty of
  *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  *    GNU General Public License for more details.
  *
  *    You should have received a copy of the GNU General Public License
  *    along with this program. If not, see <http://www.gnu.org/licenses/>.
  *    
  */
 
 import java.util.ArrayList;
 import java.util.LinkedList;

import moa.clusterers.clustree.util.*;
 import moa.cluster.Clustering;
 import moa.clusterers.AbstractClusterer;
 import moa.core.Measurement;
 import moa.options.IntOption;
 import weka.core.Instance;
 
 public class ClusTree extends AbstractClusterer{
         private static final long serialVersionUID = 1L;
         
         public IntOption horizonOption = new IntOption("horizon",
                         'h', "Range of the window.", 1000);
 
     public IntOption maxHeightOption = new IntOption(
                         "maxHeight", 'H',
                         "The maximal height of the tree", getDefaultHeight());
     
     protected int getDefaultHeight() {
         return 8;
     }
     
     private static int INSERTIONS_BETWEEN_CLEANUPS = 10000;
     protected Node root;
     // Information about the data represented in this tree.
     private int numberDimensions;
     protected double negLambda;
     private int height;
     protected int maxHeight;
     private int numRootSplits;
     private double weightThreshold = 0.05;
     private int numberInsertions;
     private long timestamp;
 
     //TODO: Add Option for that
     protected boolean breadthFirstStrat = true;
     
     //TODO: cleanup
     private Entry alsoUpdate;
     
     @Override
     public void resetLearningImpl() {
         negLambda = (1.0 / (double) horizonOption.getValue())
                 * (Math.log(weightThreshold) / Math.log(2));
         maxHeight = maxHeightOption.getValue();
         numberDimensions = -1;
         root = null;
         timestamp = 0;
         height = 0;
         numRootSplits = 0;
         numberInsertions = 0;
     }
 
 
     @Override
     protected Measurement[] getModelMeasurementsImpl() {
         return null;
     }
 
     public boolean isRandomizable() {
         return false;
     }
 
     @Override
     public void getModelDescription(StringBuilder out, int indent) {
     }
 
     public double[] getVotesForInstance(Instance inst) {
         return null;
     }
 
     @Override
     public boolean implementsMicroClusterer() {
         return true;
     }
 
     @Override
     public void trainOnInstanceImpl(Instance instance) {
         timestamp++;
         
         //TODO check if instance contains label
         if(root == null){
             numberDimensions = instance.numAttributes();
             root = new Node(numberDimensions, 0);
         }
         else{
             if(numberDimensions!=instance.numAttributes())
                 System.out.println("Wrong dimensionality, expected:"+numberDimensions+ "found:"+instance.numAttributes());
         }
 
         ClusKernel newPointAsKernel = new ClusKernel(instance.toDoubleArray(), numberDimensions);
         insert(newPointAsKernel, new SimpleBudget(1000),timestamp);
     }
 
 
     public void insert(ClusKernel newPoint, Budget budget, long timestamp) {
         if (breadthFirstStrat){
                 insertBreadthFirst(newPoint, budget, timestamp);
        }
         else{
                 Entry rootEntry = new Entry(this.numberDimensions,
                         root, timestamp, null, null);
                 ClusKernel carriedBuffer = new ClusKernel(this.numberDimensions);
                 Entry toInsertHere = insert(newPoint, carriedBuffer, root, rootEntry,
                         budget, timestamp);
         
                 if (toInsertHere != null) {
                     this.numRootSplits++;
                     this.height += this.height < this.maxHeight ? 1 : 0;
                     
                     Node newRoot = new Node(this.numberDimensions,
                             toInsertHere.getChild().getRawLevel() + 1);
                     newRoot.addEntry(rootEntry, timestamp);
                     newRoot.addEntry(toInsertHere, timestamp);
                     rootEntry.setNode(newRoot);
                     toInsertHere.setNode(newRoot);
                     this.root = newRoot;
                 }
         }
 
         this.numberInsertions++;
         if (this.numberInsertions % INSERTIONS_BETWEEN_CLEANUPS == 0) {
             cleanUp(this.root, 0);
         }
     }
 
         private Entry insertBreadthFirst(ClusKernel newPoint, Budget budget, long timestamp) {
         //check all leaf nodes and get the one with the closest entry to newPoint
                 Node bestFit = findBestLeafNode(newPoint);
         bestFit.makeOlder(timestamp, negLambda);
         Entry parent = bestFit.getEntries()[0].getParentEntry();        
         // Search for an Entry with a weight under the threshold.
             Entry irrelevantEntry = bestFit.getIrrelevantEntry(this.weightThreshold);
         int numFreeEntries = bestFit.numFreeEntries();
         Entry newEntry = new Entry(newPoint.getCenter().length,
                         newPoint, timestamp, parent, bestFit);
         //if there is space, add it to the node ( doesn't ever occur, since nodes are created with 3 entries) 
         if (numFreeEntries>0){
                 bestFit.addEntry(newEntry, timestamp);
         }
         //if outdated cluster in this best fitting node, replace it
          else if (irrelevantEntry != null) {
                 irrelevantEntry.overwriteOldEntry(newEntry);
         }
         //if there is space/outdated cluster on path to top, split. Else merge without split
         else {
                 if (existsOutdatedEntryOnPath(bestFit)||!this.hasMaximalSize()){
                     // We have to split.
                         insertHereWithSplit(newEntry, bestFit, timestamp);
                 }
                 else {
                     mergeEntryWithoutSplit(bestFit, newEntry,
                             timestamp);
                 }
         }
         //update all nodes on path to top.
         if (bestFit.getEntries()[0].getParentEntry()!=null)
                 updateToTop(bestFit.getEntries()[0].getParentEntry().getNode());
         return null;
     }
         private boolean existsOutdatedEntryOnPath(Node node) {
         if (node == root){
                 node.makeOlder(timestamp, negLambda);
                 return node.getIrrelevantEntry(this.weightThreshold)!=null;
         }
         do{
                 node = node.getEntries()[0].getParentEntry().getNode();
                 node.makeOlder(timestamp, negLambda);
                 for (Entry e : node.getEntries()){
                         e.recalculateData();
                 }
                 if (node.numFreeEntries()>0)
                         return true;
                 if (node.getIrrelevantEntry(this.weightThreshold)!=null)
                         return true;
         }while(node.getEntries()[0].getParentEntry()!=null);
                 return false;
         }
     
         private void updateToTop(Node toUpdate) {
         while(toUpdate!=null){
                 for (Entry e: toUpdate.getEntries())
                         e.recalculateData();
                 if (toUpdate.getEntries()[0].getParentEntry()==null)
                         break;
                 toUpdate=toUpdate.getEntries()[0].getParentEntry().getNode();
         }
         }
 
         private Entry insertHereWithSplit(Entry toInsert, Node insertNode,
                         long timestamp) {
                 //Handle root split
             if (insertNode.getEntries()[0].getParentEntry()==null){
                 root.makeOlder(timestamp, negLambda);
                 Entry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
                 int numFreeEntries = insertNode.numFreeEntries();
                 if (irrelevantEntry != null) {
                         irrelevantEntry.overwriteOldEntry(toInsert);
                 }
                 else if (numFreeEntries>0){
                         insertNode.addEntry(toInsert, timestamp);
                 }
                 else{
                     this.numRootSplits++;
                     this.height += this.height < this.maxHeight ? 1 : 0;
                     Entry oldRootEntry = new Entry(this.numberDimensions,
                             root, timestamp, null, null);
                     Node newRoot = new Node(this.numberDimensions,
                             this.height);
                     Entry newRootEntry = split(toInsert, root, oldRootEntry, timestamp);
                     newRoot.addEntry(oldRootEntry, timestamp);
                     newRoot.addEntry(newRootEntry, timestamp);
                     this.root = newRoot;
                     for (Entry c : oldRootEntry.getChild().getEntries())
                         c.setParentEntry(root.getEntries()[0]);
                     for (Entry c : newRootEntry.getChild().getEntries())
                         c.setParentEntry(root.getEntries()[1]);
                 }
             return null;
             }
             insertNode.makeOlder(timestamp, negLambda);
         Entry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
         int numFreeEntries = insertNode.numFreeEntries();
         if (irrelevantEntry != null) {
                 irrelevantEntry.overwriteOldEntry(toInsert);
         }
         else if (numFreeEntries>0){
                 insertNode.addEntry(toInsert, timestamp);
         }
         else {
                     // We have to split.
                         Entry parentEntry = insertNode.getEntries()[0].getParentEntry();
                         Entry residualEntry = split(toInsert, insertNode, parentEntry, timestamp);
                         if (alsoUpdate!=null){
                                 alsoUpdate = residualEntry;
                         }
                         Node nodeForResidualEntry = insertNode.getEntries()[0].getParentEntry().getNode();
                         //recursive call
                         return insertHereWithSplit(residualEntry, nodeForResidualEntry, timestamp);
             }
         
         //no Split
         return null;
         }
 
 
         // XXX: Document the insertion when the final implementation is done.
         private Entry insertHere(Entry newEntry, Node currentNode,
                 Entry parentEntry, ClusKernel carriedBuffer, Budget budget,
                 long timestamp) {
         
             int numFreeEntries = currentNode.numFreeEntries();
         
             // Insert the buffer that we carry.
             if (!carriedBuffer.isEmpty()) {
                 Entry bufferEntry = new Entry(this.numberDimensions,
                         carriedBuffer, timestamp, parentEntry, currentNode);
         
                 if (numFreeEntries <= 1) {
                     // Distance from buffer to entries.
                     Entry nearestEntryToCarriedBuffer =
                             currentNode.nearestEntry(newEntry);
                     double distanceNearestEntryToBuffer =
                             nearestEntryToCarriedBuffer.calcDistance(newEntry);
         
                     // Distance between buffer and point to insert.
                     double distanceBufferNewEntry =
                             newEntry.calcDistance(carriedBuffer);
         
                     // Best distance between Entrys in the Node.
                     BestMergeInNode bestMergeInNode =
                             calculateBestMergeInNode(currentNode);
         
                     // See what the minimal distance is and do the correspoding
                     // action.
                     if (distanceNearestEntryToBuffer <= distanceBufferNewEntry
                             && distanceNearestEntryToBuffer <= bestMergeInNode.distance) {
                         // Aggregate buffer entry to nearest entry in node.
                         nearestEntryToCarriedBuffer.aggregateEntry(bufferEntry,
                                 timestamp, this.negLambda);
                     } else if (distanceBufferNewEntry <= distanceNearestEntryToBuffer
                             && distanceBufferNewEntry <= bestMergeInNode.distance) {
                         newEntry.mergeWith(bufferEntry);
                     } else {
                         currentNode.mergeEntries(bestMergeInNode.entryPos1,
                                 bestMergeInNode.entryPos2);
                         currentNode.addEntry(bufferEntry, timestamp);
                     }
         
                 } else {
                     assert (currentNode.isLeaf());
                     currentNode.addEntry(bufferEntry, timestamp);
                 }
             }
         
             // Normally the insertion of the carries buffer does not change the
             // number of free entries, but in case of future changes we calculate
             // the number again.
             numFreeEntries = currentNode.numFreeEntries();
         
             // Search for an Entry with a weight under the threshold.
             Entry irrelevantEntry = currentNode.getIrrelevantEntry(this.weightThreshold);
             if (currentNode.isLeaf() && irrelevantEntry != null) {
                 irrelevantEntry.overwriteOldEntry(newEntry);
             } else if (numFreeEntries >= 1) {
                 currentNode.addEntry(newEntry, timestamp);
             } else {
                 if (currentNode.isLeaf() && (this.hasMaximalSize()
                         || !budget.hasMoreTime())) {
                     mergeEntryWithoutSplit(currentNode, newEntry,
                             timestamp);
                 } else {
                     // We have to split.
                     return split(newEntry, currentNode, parentEntry, timestamp);
                 }
             }
         
             return null;
         }
 
         private Node findBestLeafNode(ClusKernel newPoint) {
         double minDist = Double.MAX_VALUE;
         Node bestFit = null;
         for (Node e: collectLeafNodes(root)){
                 if (newPoint.calcDistance(e.nearestEntry(newPoint).getData())<minDist){
                         bestFit = e;
                         minDist = newPoint.calcDistance(e.nearestEntry(newPoint).getData());
                 }
         }
         if (bestFit!=null)
                 return bestFit;
         else
                 return root;
         }
     
     private ArrayList<Node> collectLeafNodes(Node curr){
         ArrayList<Node> toReturn = new ArrayList<Node>();
         if (curr==null)
                 return toReturn;
         if      (curr.isLeaf()){
                 toReturn.add(curr);
                 return toReturn;
         }
         else{
                 for (Entry e : curr.getEntries())
                         toReturn.addAll(collectLeafNodes(e.getChild()));
                 return toReturn;
         }
     }
 
         // TODO: Expand all function that work on entries to work with the Budget.
     private Entry insert(ClusKernel pointToInsert, ClusKernel carriedBuffer,
             Node currentNode, Entry parentEntry, Budget budget, long timestamp) {
         assert (currentNode != null);
         assert (currentNode.isLeaf()
                 || currentNode.getEntries()[0].getChild() != null);
 
         currentNode.makeOlder(timestamp, this.negLambda);
 
         // This variable will be changed from to null to an actual reference
         // in the following if-else block if we have to insert something here,
         // either because this is a leaf, or because of split propagation.
         Entry toInsertHere = null;
 
         if (currentNode.isLeaf()) {
             // At the end of the function the entry will be inserted.
             toInsertHere = new Entry(this.numberDimensions,
                     pointToInsert, timestamp, parentEntry, currentNode);
         } else {
 
             Entry bestEntry = currentNode.nearestEntry(pointToInsert);
             bestEntry.aggregateCluster(pointToInsert, timestamp,
                     this.negLambda);
 
             boolean isCarriedBufferEmpty = carriedBuffer.isEmpty();
 
             Entry bestBufferEntry = null;
             if (!isCarriedBufferEmpty) {
                 bestBufferEntry = currentNode.nearestEntry(carriedBuffer);
                 bestBufferEntry.aggregateCluster(carriedBuffer, timestamp,
                         this.negLambda);
             }
 
             if (!budget.hasMoreTime()) {
                 bestEntry.aggregateToBuffer(pointToInsert, timestamp,
                         this.negLambda);
                 if (!isCarriedBufferEmpty) {
                     bestBufferEntry.aggregateToBuffer(carriedBuffer,
                             timestamp, this.negLambda);
                 }
                 return null;
             }
 
             // If the way of the buffer differs from the way of the point to
             // be inserted, leave the buffer here.
             if (!isCarriedBufferEmpty && (bestEntry != bestBufferEntry)) {
                 bestBufferEntry.aggregateToBuffer(carriedBuffer, timestamp,
                         this.negLambda);
                 carriedBuffer.clear();
             }
             // Take the buffer of the best entry for the point to be inserted
             // along.
             ClusKernel takeAlongBuffer = bestEntry.emptyBuffer(timestamp,
                     this.negLambda);
             carriedBuffer.add(takeAlongBuffer);
 
             // Recursive call.
             toInsertHere = insert(pointToInsert, carriedBuffer,
                     bestEntry.getChild(), bestEntry, budget, timestamp);
         }
 
         // If the above block has a new Entry for this place insert it.
         if (toInsertHere != null) {
             return this.insertHere(toInsertHere, currentNode, parentEntry,
                     carriedBuffer, budget, timestamp);
         }
 
         // If nothing else needs to be done in all the above levels
         // return null to signalize it.
         return null;
     }
 
     private void mergeEntryWithoutSplit(Node node,
             Entry newEntry, long timestamp) {
 
         Entry nearestEntryToCarriedBuffer =
                 node.nearestEntry(newEntry);
         double distanceNearestEntryToBuffer =
                 nearestEntryToCarriedBuffer.calcDistance(newEntry);
 
         BestMergeInNode bestMergeInNode =
                 calculateBestMergeInNode(node);
 
         if (distanceNearestEntryToBuffer < bestMergeInNode.distance) {
             nearestEntryToCarriedBuffer.aggregateEntry(newEntry, timestamp,
                     this.negLambda);
         } else {
             node.mergeEntries(bestMergeInNode.entryPos1,
                     bestMergeInNode.entryPos2);
             node.addEntry(newEntry, timestamp);
         }
     }
 
     private BestMergeInNode calculateBestMergeInNode(Node node) {
         assert (node.numFreeEntries() == 0);
 
         Entry[] entries = node.getEntries();
 
         int toMerge1 = -1;
         int toMerge2 = -1;
         double distanceBetweenMergeEntries = Double.NaN;
 
         double minDistance = Double.MAX_VALUE;
         for (int i = 0; i < entries.length; i++) {
             Entry e1 = entries[i];
             for (int j = i + 1; j < entries.length; j++) {
                 Entry e2 = entries[j];
                 double distance = e1.calcDistance(e2);
                 if (distance < minDistance) {
                     toMerge1 = i;
                     toMerge2 = j;
                     distanceBetweenMergeEntries = distance;
                 }
             }
         }
 
         assert (toMerge1 != -1 && toMerge2 != -1);
         if (Double.isNaN(distanceBetweenMergeEntries)) {
             throw new RuntimeException("The minimal distance between two "
                     + "Entrys in a Node was Double.MAX_VAUE. That can hardly "
                     + "be right.");
         }
 
         return new BestMergeInNode(toMerge1, toMerge2,
                 distanceBetweenMergeEntries);
     }
 
     private boolean hasMaximalSize() {
         // TODO: Improve hasMaximalSize(). For now it just works somehow for testing.
         return this.height == this.maxHeight;
     }
 
     private Entry split(Entry newEntry, Node node, Entry parentEntry,
             long timestamp) {
         // The implemented split function only works in trees where node
         // have three entries.
         // Splitting only makes sense on full nodes.
         assert (node.numFreeEntries() == 0);
         assert (parentEntry.getChild() == node);
 
         // All the entries we have to separate in two nodes.
         Entry[] allEntries = new Entry[4];
         Entry[] nodeEntries = node.getEntries();
         for (int i = 0; i < nodeEntries.length; i++) {
             allEntries[i] = new Entry(nodeEntries[i]);
         }
         allEntries[3] = newEntry;
 
         // Clear the given node, since we are going to refill it later.
         node = new Node(this.numberDimensions, node.getRawLevel());
 
         // Calculate the distance of all the possible pairings, since we want
         // to do a (2,2) split.
         double select01 = allEntries[0].calcDistance(allEntries[1])
                 + allEntries[2].calcDistance(allEntries[3]);
 
         double select02 = allEntries[0].calcDistance(allEntries[2])
                 + allEntries[1].calcDistance(allEntries[3]);
 
         double select03 = allEntries[0].calcDistance(allEntries[3])
                 + allEntries[1].calcDistance(allEntries[2]);
 
         // See which of the pairings is minimal and distribute the entries
         // accordingly.
         Node residualNode = new Node(this.numberDimensions,
                 node.getRawLevel());
         if (select01 < select02) {
             if (select01 < select03) {//select01 smallest
                 node.addEntry(allEntries[0], timestamp);
                 node.addEntry(allEntries[1], timestamp);
                 residualNode.addEntry(allEntries[2], timestamp);
                 residualNode.addEntry(allEntries[3], timestamp);
             } else {//select03 smallest
                 node.addEntry(allEntries[0], timestamp);
                 node.addEntry(allEntries[3], timestamp);
                 residualNode.addEntry(allEntries[1], timestamp);
                 residualNode.addEntry(allEntries[2], timestamp);
             }
         } else {
             if (select02 < select03) {//select02 smallest
                 node.addEntry(allEntries[0], timestamp);
                 node.addEntry(allEntries[2], timestamp);
                 residualNode.addEntry(allEntries[1], timestamp);
                 residualNode.addEntry(allEntries[3], timestamp);
             } else {//select03 smallest
                 node.addEntry(allEntries[0], timestamp);
                 node.addEntry(allEntries[3], timestamp);
                 residualNode.addEntry(allEntries[1], timestamp);
                 residualNode.addEntry(allEntries[2], timestamp);
             }
         }
 
         // Set the other node into the tree.
         parentEntry.setChild(node);
         parentEntry.recalculateData();
         int count = 0;
         for (Entry e : node.getEntries()){
                 e.setParentEntry(parentEntry);
                 if (e.getData().getN() != 0)
                         count++;
         }
         //System.out.println(count);
         // Generate a new entry for the residual node.
         Entry residualEntry = new Entry(this.numberDimensions,
                 residualNode, timestamp, parentEntry, node);
         count=0;
         for (Entry e: residualNode.getEntries()){
                 e.setParentEntry(residualEntry);
                 if (e.getData().getN() != 0)
                         count++;
         }
         //System.out.println(count);
         return residualEntry;
     }
 
     public int getNumRootSplits() {
         return numRootSplits;
     }
 
     public int getHeight() {
         assert (height <= maxHeight);
         return height;
     }
 
     private void cleanUp(Node currentNode, int level) {
         if (currentNode == null) {
             return;
         }
 
         Entry[] entries = currentNode.getEntries();
         if (level == this.maxHeight) {
             for (int i = 0; i < entries.length; i++) {
                 Entry e = entries[i];
                 e.setChild(null);
             }
         } else {
             for (int i = 0; i < entries.length; i++) {
                 Entry e = entries[i];
                 cleanUp(e.getChild(), level + 1);
             }
         }
     }
 
     //TODO: Microcluster unter dem Threshhold nich zur�ckgeben (WIe bei outdated entries)
     @Override
     public Clustering getMicroClusteringResult() {
         return getClustering(timestamp, -1);
     }
 
     @Override
     public Clustering getClusteringResult() {
         return null;
     }
 
 
     public Clustering getClustering(long currentTime, int targetLevel) {
         if (root == null) {
             return null;
         }
 
         Clustering clusters = new Clustering();
         LinkedList<Node> queue = new LinkedList<Node>();
         queue.add(root);
 
         while (!queue.isEmpty()) {
             Node current = queue.remove();
            // if (current == null)
            //   continue;
             int currentLevel = current.getLevel(this);
             boolean isLeaf = (current.isLeaf() && currentLevel <= maxHeight)
                     || currentLevel == maxHeight;
 
             if (currentLevel == targetLevel
                     || (targetLevel == - 1 && isLeaf)) {
                 assert (currentLevel <= maxHeight);
 
                 Entry[] entries = current.getEntries();
                 for (int i = 0; i < entries.length; i++) {
                     Entry entry = entries[i];
                     if (entry == null || entry.isEmpty()) {
                         continue;
                     }
                     // XXX
                     entry.makeOlder(currentTime, this.negLambda);
                     if (entry.isIrrelevant(this.weightThreshold))
                         continue;
 
                     ClusKernel gaussKernel = new ClusKernel(entry.getData());
 
 //                  long diff = currentTime - entry.getTimestamp();
 //                    if (diff > 0) {
 //                        gaussKernel.makeOlder(diff, negLambda);
 //                    }
 
                     clusters.add(gaussKernel);
                 }
             } else if (!current.isLeaf()) {
                 Entry[] entries = current.getEntries();
                 for (int i = 0; i < entries.length; i++) {
                     Entry entry = entries[i];
 
                     if (entry.isEmpty()) {
                         continue;
                     }
 
                     if (entry.isIrrelevant(weightThreshold)) {
                         continue;
                     }
 
                     queue.add(entry.getChild());
                 }
             }
         }
 
         return clusters;
     }
 
 
 
     /**************************************************************************
      * LOCAL CLASSES
      **************************************************************************/
     class BestMergeInNode {
 
         public int entryPos1;
         public int entryPos2;
         public double distance;
 
         public BestMergeInNode(int pos1, int pos2,
                 double distance) {
             assert (pos1 != pos2);
 
             this.distance = distance;
 
             if (pos1 < pos2) {
                 this.entryPos1 = pos1;
                 this.entryPos2 = pos2;
             } else {
                 this.entryPos1 = pos2;
                 this.entryPos2 = pos1;
             }
         }
     }
 
 }

