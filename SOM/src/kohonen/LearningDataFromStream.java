package kohonen;

import java.util.ArrayList;

public class LearningDataFromStream implements LearningDataModel{

	/**
     * ArrayList contains learning data
     */
    private ArrayList <double[]> dataList = new ArrayList<double[]>();
	
	public void addData(double[] dataPoint){
		dataList.add(dataPoint);
	}
	
	@Override
	public double[] getData(int index) {
		return dataList.get(index);
	}

	@Override
	public int getDataSize() {
		return dataList.size();
	}

	/**
	 * Clear all the learning data.
	 */
	public void clear() {
		dataList.clear();		
	}

}
