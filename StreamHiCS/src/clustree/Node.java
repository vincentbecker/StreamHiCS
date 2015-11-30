package clustree;

/*
 *    Node.java
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

public class Node {

	final int NUMBER_ENTRIES = 3;
	static int INSERTIONS_BETWEEN_CLEANUPS = 10000;

	private Entry[] entries;
	// Information about the state in the tree.
	private int level;

	public Node(int numberDimensions, int level) {
		this.level = level;

		this.entries = new Entry[NUMBER_ENTRIES];
		// Generate all entries when we generate a Node.
		// That no entry can be null makes it much easier to handle.
		for (int i = 0; i < NUMBER_ENTRIES; i++) {
			entries[i] = new Entry(numberDimensions);
			entries[i].setNode(this);
		}
	}

	protected Node(int numberDimensions, int numberClasses, int level, boolean fakeRoot) {

		this.level = level;

		this.entries = new Entry[NUMBER_ENTRIES];
		// Generate all entries when we generate a Node.
		// That no entry can be null makes it much easier to handle.
		for (int i = 0; i < NUMBER_ENTRIES; i++) {
			entries[i] = new Entry(numberDimensions);
		}
	}

	public Node(int numberDimensions, int level, Entry[] argEntries) {
		this.level = level;

		this.entries = new Entry[NUMBER_ENTRIES];
		// Generate all entries when we generate a Node.
		// That no entry can be null makes it much easier to handle.
		for (int i = 0; i < NUMBER_ENTRIES; i++) {
			entries[i] = argEntries[i];
		}
	}

	protected boolean isLeaf() {

		for (int i = 0; i < entries.length; i++) {
			Entry entry = entries[i];
			if (entry.getChild() != null) {
				return false;
			}
		}

		return true;
	}

	public Entry nearestEntry(ClusKernel buffer) {

		// TODO: (Fernando) Adapt the nearestEntry(...) function to the new
		// algorithmn.

		Entry res = entries[0];
		double min = res.calcDistance(buffer);
		for (int i = 1; i < entries.length; i++) {
			Entry entry = entries[i];

			if (entry.isEmpty()) {
				break;
			}

			double distance = entry.calcDistance(buffer);
			if (distance < min) {
				min = distance;
				res = entry;
			}
		}

		return res;
	}

	protected Entry nearestEntry(Entry newEntry) {
		assert (!this.entries[0].isEmpty());

		Entry res = entries[0];
		double min = res.calcDistance(newEntry);
		for (int i = 1; i < entries.length; i++) {

			if (this.entries[i].isEmpty()) {
				break;
			}

			Entry entry = entries[i];
			double distance = entry.calcDistance(newEntry);
			if (distance < min) {
				min = distance;
				res = entry;
			}
		}

		return res;
	}

	protected int numFreeEntries() {
		int res = 0;
		for (int i = 0; i < entries.length; i++) {
			Entry entry = entries[i];
			if (entry.isEmpty()) {
				res++;
			}
		}

		assert (NUMBER_ENTRIES == entries.length);
		return res;
	}

	public void addEntry(Entry newEntry, long currentTime) {
		newEntry.setNode(this);
		int freePosition = getNextEmptyPosition();
		entries[freePosition].initializeEntry(newEntry, currentTime);
	}

	private int getNextEmptyPosition() {
		int counter;
		for (counter = 0; counter < entries.length; counter++) {
			Entry e = entries[counter];
			if (e.isEmpty()) {
				break;
			}
		}

		if (counter == entries.length) {
			throw new RuntimeException("Entry added to a node which is already full.");
		}

		return counter;
	}

	protected Entry getIrrelevantEntry(double threshold) {
		for (int i = 0; i < this.entries.length; i++) {
			Entry entry = this.entries[i];
			if (entry.isIrrelevant(threshold)) {
				return entry;
			}
		}

		return null;
	}

	public Entry[] getEntries() {
		return entries;
	}

	protected int getRawLevel() {
		return level;
	}

	protected int getLevel(ClusTree tree) {
		int numRootSplits = tree.getNumRootSplits();
		return numRootSplits - this.getRawLevel();
	}

	// Level stays the same.
	protected void clear() {
		for (int i = 0; i < NUMBER_ENTRIES; i++) {
			entries[i].shallowClear();
		}
	}

	protected void mergeEntries(int pos1, int pos2) {
		assert (this.numFreeEntries() == 0);
		assert (pos1 < pos2);

		this.entries[pos1].mergeWith(this.entries[pos2]);

		for (int i = pos2; i < entries.length - 1; i++) {
			entries[i] = entries[i + 1];
		}
		entries[entries.length - 1].clear();
	}

	protected void makeOlder(long currentTime, double negLambda) {
		for (int i = 0; i < this.entries.length; i++) {
			Entry entry = this.entries[i];

			if (entry.isEmpty()) {
				break;
			}

			entry.makeOlder(currentTime, negLambda);
		}
	}
}