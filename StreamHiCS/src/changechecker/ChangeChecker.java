package changechecker;

public abstract class ChangeChecker {

	private int time = 0;
	private int checkInterval;
	
	public ChangeChecker(int checkInterval){
		this.checkInterval = checkInterval;
	}
	
	public boolean poll(){
		time++;
		if(time % checkInterval == 0){
			time = 0;
			return true;
		}
		return false;
	}
	
	public abstract boolean checkForChange();
}
