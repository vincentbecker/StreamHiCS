import kohonen.LearningDataFromStream;
import kohonen.WTMLearningFunction;
import learningFactorFunctional.HiperbolicFunctionalFactor;
import metrics.EuclidesMetric;
import network.DefaultNetwork;
import topology.GaussNeighbourhoodFunction;
import topology.MatrixTopology;

public class Main {
	public static void main(String[] args) {
		MatrixTopology topology = new MatrixTopology(30, 30);
		double[] maxWeight = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
		DefaultNetwork network = new DefaultNetwork(11, maxWeight, topology);
		LearningDataFromStream fileData = new LearningDataFromStream();
		HiperbolicFunctionalFactor learningFunction = new HiperbolicFunctionalFactor(1, 1);
		WTMLearningFunction learning = new WTMLearningFunction(network, 20, new EuclidesMetric(), fileData,
				learningFunction, new GaussNeighbourhoodFunction(2));
		learning.learn();
		System.out.println(network);
	}
}
