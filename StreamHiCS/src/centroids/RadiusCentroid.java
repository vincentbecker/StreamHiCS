package centroids;

public class RadiusCentroid extends Centroid{
	private static final double RADIUSMULT = 1.8;
	private double[] LS;
	private double[] SS;
	private double maxRadius;

	public RadiusCentroid(double[] vector,  double negLambda, int currentTime, double maxRadius) {
		super(negLambda, currentTime);
		this.LS = vector;
		int l = LS.length;
		this.SS = new double[l];
		for (int i = 0; i < l; i++) {
			SS[i] = Math.pow(LS[i], 2);
		}
		this.maxRadius = maxRadius;
	}

	@Override
	public boolean addPointImpl(double[] point) {
		
		double max = 0;
		double r = 0;
		for (int i = 0; i < LS.length; i++) {
			r = Math.sqrt(SS[i] / weight - Math.pow(LS[i] / weight, 2));
			if(r > max){
				max = r;
			}
		}
		
		//return 1.8 * max;
		
		double val;
		for(int i = 0; i < LS.length; i++){
			val = point[i];
			LS[i] += val;
			SS[i] += Math.pow(val, 2);
		}
		
		double distance = this.euclideanDistance(point);
		
		return true;
	}
	
	@Override
	public void fadeImpl(int currentTime) {
		double fadingFactor = Math.pow(2.0, negLambda * (currentTime - lastUpdate));
		for (int i = 0; i < LS.length; i++) {
			LS[i] = LS[i] * fadingFactor;
			SS[i] = SS[i] * fadingFactor;
		}
	}

	public double[] getCentre() {
		int l = LS.length;
		double[] centre = new double[l];
		for (int i = 0; i < l; i++) {
			centre[i] = LS[i] / weight;
		}
		return centre;
	}
	
	@Override
	public double getRadiusImpl() {
		double max = 0;
		double r = 0;
		for (int i = 0; i < LS.length; i++) {
			r = Math.sqrt(SS[i] / weight - Math.pow(LS[i] / weight, 2));
			if(r > max){
				max = r;
			}
		}
		
		return RADIUSMULT * max;
	}

	public void fade(int currentTime) {
		double fadingFactor = Math.pow(2.0, negLambda * (currentTime - lastUpdate));
		weight = weight * fadingFactor;
		for (int i = 0; i < LS.length; i++) {
			LS[i] = LS[i] * fadingFactor;
			SS[i] = SS[i] * fadingFactor;
		}
		lastUpdate = currentTime;
	}

	
}
