package StreamDataStructures;
import java.util.ArrayList;

public class Subspace {

	private ArrayList<Integer> dimensions;
	
	public ArrayList<Integer> getDimensions() {
		return dimensions;
	}
	
	public int getSize(){
		return dimensions.size();
	}

	public int getDimension(int index) {
		return dimensions.get(index);
	}
}
