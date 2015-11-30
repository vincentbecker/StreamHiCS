package clustree;
 /*
  *    Entry.java
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

public class Entry {
 
     public ClusKernel data;
     private ClusKernel buffer;
     private Node child;
     private Entry parentEntry;
     private Node node;
     private long timestamp;
     private static final long defaultTimestamp = 0;
 
     public Entry(int numberDimensions) {
         this.data = new ClusKernel(numberDimensions);
         this.buffer = new ClusKernel(numberDimensions);
         this.child = null;
         this.timestamp = Entry.defaultTimestamp;
     }
 
     protected Entry(int numberDimensions,
             Node node, long currentTime, Entry parentEntry, Node containerNode) {
         this(numberDimensions);
         this.child = node;
         this.parentEntry = parentEntry;
         this.node = containerNode;
         Entry[] entries = node.getEntries();
         for (int i = 0; i < entries.length; i++) {
             Entry entry = entries[i];
             entry.setParentEntry(this);
             if (entry.isEmpty()) {
                 break;
             }
 
             this.add(entry);
         }
 
         this.timestamp = currentTime;
     }
 
 
 
     public Entry(int numberDimensions, ClusKernel cluster, long currentTime) {
         this(numberDimensions);
         this.data.add(cluster);
         this.timestamp = currentTime;
     }
     protected Entry(int numberDimensions, ClusKernel cluster, long currentTime, Entry parentEntry, Node containerNode) {
         this(numberDimensions);
         this.parentEntry = parentEntry;
         this.data.add(cluster);
         this.node = containerNode;
         this.timestamp = currentTime;
     }
     protected Entry(Entry other) {
         this.parentEntry = other.parentEntry;
         this.node = other.node;
         this.buffer = new ClusKernel(other.buffer);
         this.data = new ClusKernel(other.data);
         this.timestamp = other.timestamp;
         this.child = other.child;
         if (other.getChild()!=null)
                 for (Entry e : other.getChild().getEntries()){
                         e.setParentEntry(this);
                 }
     }
 
     public Node getNode() {
                 return node;
         }
 
         public void setNode(Node node) {
                 this.node = node;
         }
 
     protected void clear() {
         this.data.clear();
         this.buffer.clear();
         this.child = null;
         this.timestamp = Entry.defaultTimestamp;
     }
 
     protected void shallowClear() {
         this.buffer.clear();
         this.data.clear();
     }
 
     protected double calcDistance(ClusKernel cluster) {
         return data.calcDistance(cluster);
     }
 
     public double calcDistance(Entry other) {
         return this.getData().calcDistance(other.getData());
     }
 
     protected void initializeEntry(Entry other, long currentTime) {
         assert (this.isEmpty());
         assert (other.getBuffer().isEmpty());
         this.data.add(other.data);
         this.timestamp = currentTime;
         this.child = other.child;
         if (child!=null){
                 for (Entry e : child.getEntries()){
                         e.setParentEntry(this);
                 }
         }
     }
 
     public void add(Entry other) {
         this.data.add(other.data);
     }
 
     protected void aggregateEntry(Entry other, long currentTime,
             double negLambda) {
         this.data.aggregate(other.data, currentTime - this.timestamp,
                 negLambda);
         this.timestamp = currentTime;
     }
 
     protected void aggregateCluster(ClusKernel otherData, long currentTime,
             double negLambda) {
         this.getData().aggregate(otherData, currentTime - this.timestamp,
                 negLambda);
         this.timestamp = currentTime;
     }
 
     protected void aggregateToBuffer(ClusKernel pointToInsert, long currentTime,
             double negLambda) {
         ClusKernel currentBuffer = this.getBuffer();
         currentBuffer.aggregate(pointToInsert, currentTime - this.timestamp,
                 negLambda);
         this.timestamp = currentTime;
     }
 
     protected void mergeWith(Entry other) {
         // We should only merge entries in leafs, and leafes should have empty
         // buffers.
         assert (this.child == null);
         assert (other.child == null);
         assert (other.buffer.isEmpty());
 
         this.data.add(other.data);
         if (this.timestamp < other.timestamp) {
             this.timestamp = other.timestamp;
         }
     }
 
     protected ClusKernel getBuffer() {
         return buffer;
     }
 
     public Node getChild() {
         return child;
     }
 
     protected ClusKernel getData() {
         return data;
     }
     public Entry getParentEntry() {
                 return parentEntry;
         }
 
         public void setParentEntry(Entry parent) {
                 this.parentEntry = parent;
         }
 
     public void setChild(Node child) {
         this.child = child;
     }
 
     public long getTimestamp() {
         return timestamp;
     }
 
     protected ClusKernel emptyBuffer(long currentTime, double negLambda) {
         this.buffer.makeOlder(currentTime - this.timestamp, negLambda);
         ClusKernel bufferCopy = new ClusKernel(this.buffer);
         this.buffer.clear();
         return bufferCopy;
     }
 
     protected boolean isEmpty() {
         // Assert that if the data cluster is empty, the buffer cluster is
         // empty too.
         assert ((this.data.isEmpty() && this.buffer.isEmpty())
                 || !this.data.isEmpty());
 
         return this.data.isEmpty();
     }
 
     protected void overwriteOldEntry(Entry newEntry) {
         assert (this.getBuffer().isEmpty());
         assert (newEntry.getBuffer().isEmpty());
         this.data.overwriteOldCluster(newEntry.data);
         newEntry.setParentEntry(this.parentEntry);
         if (newEntry.getChild()!=null)
         for (Entry e : newEntry.getChild().getEntries())
                 e.setParentEntry(this);
         //this.setParentEntry(newEntry.getParentEntry());
         this.child=newEntry.child;
     }
 
     protected void recalculateData() {
         Node currentChild = this.getChild();
         if (currentChild != null) {
             ClusKernel currentData = this.getData();
             currentData.clear();
             Entry[] entries = currentChild.getEntries();
             for (int i = 0; i < entries.length; i++) {
                 currentData.add(entries[i].getData());
             }
         } else {
             this.clear();
         }
     }
 
     protected boolean isIrrelevant(double threshold) {
         return this.getData().getWeight() < threshold;
     }
 
     protected void makeOlder(long currentTime, double negLambda) {
         assert (currentTime > this.timestamp) : "currentTime : "
                 + currentTime + ", this.timestamp: " + this.timestamp;
 
         long diff = currentTime - this.timestamp;
         this.buffer.makeOlder(diff, negLambda);
         this.data.makeOlder(diff, negLambda);
         this.timestamp = currentTime;
     }
 
 }
