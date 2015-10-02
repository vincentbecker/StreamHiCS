import weka.core.Instance;
import moa.streams.generators.RandomRBFGenerator;
import streams.CorrelatedStream;
import streams.GaussianStream;

public class Main {

	public static void main(String[] args) {
		int numInstances = 100000;
		// RandomRBFGenerator stream = new RandomRBFGenerator();
		// stream.prepareForUse();
		// int numberOfDimensions = stream.numAttsOption.getValue();
		// CorrelatedStream stream = new CorrelatedStream();
		double[][] covariances = { { 1, 0.9, 0, 0, 0 }, { 0.9, 1, 0, 0, 0 }, { 0, 0, 1, 0.5, 0.3 },
				{ 0, 0, 0.5, 1, 0.3 }, { 0, 0, 0.3, 0.3, 1 } };
		GaussianStream stream = new GaussianStream(covariances);
		int numberOfDimensions = stream.getNumberOfDimensions();

		StreamHiCS streamHiCS = new StreamHiCS(numberOfDimensions, 10000, 20, 0.3, 1);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
			// System.out.println(streamHiCS.getNumberOfInstances());
		}

		/*
		 * int numInstances = 10000; RandomRBFGenerator stream = new
		 * RandomRBFGenerator(); stream.prepareForUse(); ArrayList<Attribute> a
		 * = new ArrayList<Attribute>(); for(int i = 0; i<
		 * stream.getHeader().numAttributes(); i++){
		 * a.add(stream.getHeader().attribute(i)); } SelfOrganizingMapContainer
		 * som = new SelfOrganizingMapContainer(a);
		 * 
		 * int numberSamples = 0; while (stream.hasMoreInstances() &&
		 * numberSamples < numInstances) { Instance inst =
		 * stream.nextInstance(); System.out.println(inst.value(0));
		 * som.add(inst); numberSamples++; //
		 * System.out.println(streamHiCS.getNumberOfInstances()); }
		 * 
		 * som.trainSOM(); System.out.println(som.getNumberOfInstances());
		 */
	}
}
