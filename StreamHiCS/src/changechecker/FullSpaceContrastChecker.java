package changechecker;

import contrast.Contrast;
import subspace.Subspace;

public class FullSpaceContrastChecker extends ChangeChecker {

	private Contrast contrastEvaluator;
	private Subspace fullSpace;
	private double minContrast = Double.MIN_VALUE;
	private double maxContrast = Double.MIN_VALUE;
	private double weightedAverage;
	private double gamma;
	private double threshold;
	private boolean init = false;

	public FullSpaceContrastChecker(int checkInterval, int numberOfDimensions, Contrast contrastEvaluator, double gamma, 
			double threshold) {
		super(checkInterval);
		int[] dimensions = new int[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			dimensions[i] = i;
		}
		fullSpace = new Subspace(dimensions);
		this.contrastEvaluator = contrastEvaluator;
		this.gamma = gamma;
		this.threshold = threshold;
	}

	public void setContrastEvaluator(Contrast contrastEvaluator) {
		this.contrastEvaluator = contrastEvaluator;
	}

	@Override
	public boolean checkForChange() {
		return weightedAverageMethod();
	}
	
	private boolean minMaxMethod(){
		double contrast = contrastEvaluator.evaluateSubspaceContrast(fullSpace);

		System.out.println("Contrast: " + contrast);
		
		if (!init) {
			minContrast = contrast;
			maxContrast = contrast;
			init = true;
		}

		double minDifference = minContrast - contrast;
		double maxDifference = contrast - maxContrast;

		if (contrast < minContrast) {
			minContrast = contrast;
		}
		if (contrast > maxContrast) {
			maxContrast = contrast;
		}

		// double difference = Math.abs(lastContrast - contrast);
		// lastContrast = contrast;
		System.out.println("MinDifference: " + minDifference + ", MaxDifference: " + maxDifference);
		if (minDifference > threshold || maxDifference > threshold) {
			minContrast = 0;
			maxContrast = 0;
			init = false;
			return true;
		}
		return false;
	}
	
	private boolean weightedAverageMethod(){
		double contrast = contrastEvaluator.evaluateSubspaceContrast(fullSpace);
		System.out.println("Contrast: " + contrast);
		
		double difference = contrast - weightedAverage;
		System.out.println("Difference to average: " + difference);
		
		if (Math.abs(difference) > threshold){
			weightedAverage = contrast;
			return true;
		}
		
		weightedAverage = gamma*contrast + (1-gamma)*weightedAverage;
		
		return false;
	}

}
