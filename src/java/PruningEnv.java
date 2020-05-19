// Environment code for project pruning

import jason.asSyntax.*;
import jason.environment.*;
import java.util.logging.*;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
public class PruningEnv extends Environment {

    private Logger logger = Logger.getLogger("pruning."+PruningEnv.class.getName());
    
    private Literal remainL = Literal.parseLiteral("remaingLayers");
    private Literal notRemainL = Literal.parseLiteral("~remaingLayers");
    
    private Literal undoPruning = Literal.parseLiteral("decreasePerformance");
    private Literal keepPruning = Literal.parseLiteral("~decreasePerformance");
    
    private Literal keepUntrained = Literal.parseLiteral("~trainAgain");
    private Literal trainAgain = Literal.parseLiteral("trainAgain");
    
    private Model M;
    private int counter = 0;
    private boolean continue_pruning = true;

    /** Called before the MAS execution with the args informed in .mas2j */
    @Override
    public void init(String[] args) {
        super.init(args);
        try {
        	addPercept(this.remainL);
        	addPercept(this.keepPruning);
        	addPercept(this.keepUntrained);
        } catch (Exception e){
        	System.out.println(e.getStackTrace());
        }
        this.M = new Model();
    }

    @Override
    public boolean executeAction(String agName, Structure action) {
    	System.out.println("\tAgent "+agName+" is doing "+action);
    	clearPercepts();
        
        /* Performing the actions */
        if (action.getFunctor().equals("make_prune")) {
        	prune();
//        	System.out.println("\tRealizando pruning");
        	
        } else if (action.getFunctor().equals("verify")) {
        	this.continue_pruning = evaluatePerformance();
//        	System.out.println("\tVerificando na tabela a performance");
        	
        	if (this.continue_pruning && this.counter <= 12) { // After, continue_pruning
        		System.out.println("\t1");
        		addPercept(this.remainL);
        		addPercept(this.keepPruning);
        		addPercept(this.keepUntrained);
        	} else if (!this.continue_pruning && this.counter <= 12) { // After, undo_prune
        		System.out.println("\t2");
        		addPercept(this.remainL);
        		addPercept(this.undoPruning);
        		addPercept(this.keepUntrained);
        	} else if (this.continue_pruning && this.counter > 12) { // after, train
        		System.out.println("\t3");
        		addPercept(this.remainL);
        		addPercept(this.undoPruning);
        		addPercept(this.trainAgain);
        	} else if (!this.continue_pruning && this.counter > 12) { // After, just_end
        		System.out.println("\t4");
        		addPercept(this.notRemainL);
        		try {
        			TimeUnit.MINUTES.sleep(5);
        		} catch (Exception e) {
        			
        		}
        	}
        	
        } else if (action.getFunctor().equals("train")) {
        
    	} else if (action.getFunctor().equals("undo_prune")) {
//        	System.out.println("\tDesfazendo o ultimo pruning");
        	
        } else if (action.getFunctor().equals("continue_pruning")) {
//        	System.out.println("\tMantendo o ultimo pruning");
        	
        } else if (action.getFunctor().equals("just_end")) {
        	System.out.println("\tProcesso finalizado");
        	
        }
         
        return true;
    }

    /** Called before the end of MAS execution */
    @Override
    public void stop() {
        super.stop();
    }
    
//    public boolean prune() {
//    	int i = ThreadLocalRandom.current().nextInt(0, this.M.size());
//    	Layer l = this.M.getLayer(i);
//    	
//    	int j = ThreadLocalRandom.current().nextInt(0, l.size());
//    	Item channel = l.getChannel(j);
//    	
//    	channel.nTimes += 1;
//    	channel.performance -= 5.0;
//    }
    public void prune() {
    	this.counter += 1;
    }
    
    
    public boolean evaluatePerformance() {
    	if (this.counter % 3 == 0) return false;
    	return true;
    }
    
    
    /*
     * Implementation of a model
     */
    class Model {
    	ArrayList<Layer> layers = new ArrayList();
    	
    	public Model () {
    		for (int i = 0; i < 5; i++) {
    			layers.add(new Layer(15));
    		}
    	}
    	
    	public Layer getLayer(int i ) {
    		return layers.get(i);
    	}
    	
    	public int size() {
    		return layers.size();
    	}
    }
    
    /* 
     * Implementation of a Layer
     */
    class Layer {
    	ArrayList<Item> channels = new ArrayList();
    	
    	public Layer (int nChannels) {
    		for (int i = 0; i < nChannels; i++) {
    			Item it = new Item(i, 100.0);
    			this.channels.add(it);
    		}
    	}
    	
    	public Item getChannel(int i) {
    		return this.channels.get(i);
    	}
    	
    	public int size() {
    		return channels.size();
    	}
    }
    
    class Item {
    	public int index;
    	public double performance;
    	public int nTimes;
    	
		public Item(int index, double performance) {
    		this.index = index;
    		this.performance = performance;
    		this.nTimes = 0;
    	}
    }
}

