import java.awt.EventQueue;
import visualisation.PointsEx;

public class Main {

	public static void main(String[] args) {
		// Visualisation
		
		 EventQueue.invokeLater(new Runnable() {
		 @Override public void run() {
		  
		 PointsEx ex = new PointsEx(); ex.setVisible(true); } });

		/*
		int numInstances = 10000;
		RandomRBFGenerator stream = new RandomRBFGenerator();
		stream.prepareForUse();
		int numberOfDimensions = stream.numAttsOption.getValue(); //


		 * CorrelatedStream stream = new CorrelatedStream(); double[][]
		 * covariances = { { 1, 0.9, 0, 0, 0 }, { 0.9, 1, 0, 0, 0 }, { 0, 0, 1,
		 * 0.5, 0.3 }, { 0, 0, 0.5, 1, 0.3 }, { 0, 0, 0.3, 0.3, 1 } };
		 * GaussianStream stream = new GaussianStream(covariances);


		// int numberOfDimensions = stream.getNumberOfDimensions();

		StreamHiCS streamHiCS = new StreamHiCS(numberOfDimensions, numberOfDimensions, numberOfDimensions,
				numberOfDimensions, numberOfDimensions, null);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		 * int numInstances = 1000; RandomRBFGenerator stream = new
		 * RandomRBFGenerator(); stream.prepareForUse(); SelfOrganizingMap som =
		 * new SelfOrganizingMap(stream.getHeader().numAttributes(), 25);
		 * 
		 * int numberSamples = 0; while (stream.hasMoreInstances() &&
		 * numberSamples < numInstances) { Instance inst =
		 * stream.nextInstance(); som.add(inst); numberSamples++; }
		 * 
		 * som.trainSOM(); System.out.println("Number of clusters: " +
		 * som.getNumberOfInstances());
		 */
	}
}
