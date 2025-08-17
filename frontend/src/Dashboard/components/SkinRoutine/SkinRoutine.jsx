import React, { useState, useEffect, useCallback } from 'react';
import { Sparkles, Droplets, Calendar, Apple, Lightbulb, Sunrise, Moon, CalendarDays, Sun, Sunset, Pill, DropletIcon, X, Star } from 'lucide-react';
import './SkinRoutine.css';

const SkinRoutine = ({ latestAnalysis }) => {
  const [activeTab, setActiveTab] = useState('routine');
  const [isGenerating, setIsGenerating] = useState(false);
  const [routine, setRoutine] = useState(null);
  const [dietPlan, setDietPlan] = useState(null);

  const generateRoutine = useCallback(async () => {
    if (!latestAnalysis) return;
    
    setIsGenerating(true);
    
    // Simulate API call to Gemini - replace with actual implementation
    setTimeout(() => {
      const mockRoutine = {
        skinType: latestAnalysis.skinType,
        mainConcerns: latestAnalysis.concerns,
        morning: [
          {
            step: 1,
            product: "Gentle Cleanser",
            description: "Use a mild, pH-balanced cleanser to remove overnight buildup",
            duration: "1-2 minutes",
            ingredients: ["Ceramides", "Hyaluronic Acid"]
          },
          {
            step: 2,
            product: "Vitamin C Serum",
            description: "Apply antioxidant protection for environmental damage",
            duration: "Let absorb for 5 minutes",
            ingredients: ["L-Ascorbic Acid", "Vitamin E"]
          },
          {
            step: 3,
            product: "Hydrating Moisturizer",
            description: "Lock in moisture with a lightweight, non-comedogenic formula",
            duration: "Apply generously",
            ingredients: ["Niacinamide", "Peptides"]
          },
          {
            step: 4,
            product: "Broad Spectrum SPF 30+",
            description: "Essential sun protection - reapply every 2 hours",
            duration: "Never skip!",
            ingredients: ["Zinc Oxide", "Titanium Dioxide"]
          }
        ],
        evening: [
          {
            step: 1,
            product: "Oil Cleanser",
            description: "Remove makeup and sunscreen thoroughly",
            duration: "Massage for 1 minute",
            ingredients: ["Jojoba Oil", "Emulsifiers"]
          },
          {
            step: 2,
            product: "Gentle Cleanser",
            description: "Second cleanse to remove remaining impurities",
            duration: "1-2 minutes",
            ingredients: ["Ceramides", "Hyaluronic Acid"]
          },
          {
            step: 3,
            product: "Treatment Serum",
            description: "Target specific concerns with active ingredients",
            duration: "Wait 10 minutes before next step",
            ingredients: ["Retinol", "Peptides"]
          },
          {
            step: 4,
            product: "Night Moisturizer",
            description: "Rich, repairing formula for overnight recovery",
            duration: "Apply as final step",
            ingredients: ["Ceramides", "Squalane"]
          }
        ],
        weekly: [
          {
            frequency: "2-3 times per week",
            product: "Gentle Exfoliant",
            description: "Remove dead skin cells for better product absorption",
            ingredients: ["BHA", "AHA"]
          },
          {
            frequency: "Once per week",
            product: "Hydrating Face Mask",
            description: "Deep moisture treatment for plump, healthy skin",
            ingredients: ["Hyaluronic Acid", "Aloe Vera"]
          }
        ],
        tips: [
          "Always patch test new products",
          "Introduce new products gradually",
          "Consistency is key - give products 4-6 weeks to show results",
          "Listen to your skin and adjust routine as needed"
        ]
      };
      
      setRoutine(mockRoutine);
      setIsGenerating(false);
    }, 2000);
  }, [latestAnalysis]);

  const generateDietPlan = async () => {
    if (!latestAnalysis) return;
    
    setIsGenerating(true);
    
    // Simulate API call to Gemini - replace with actual implementation
    setTimeout(() => {
      const mockDietPlan = {
        goals: latestAnalysis.concerns,
        dailyPlan: {
          breakfast: [
            {
              food: "Greek Yogurt with Berries",
              benefits: "Probiotics for gut health, antioxidants for skin protection",
              nutrients: ["Vitamin C", "Protein", "Probiotics"]
            },
            {
              food: "Avocado Toast on Whole Grain",
              benefits: "Healthy fats for skin barrier function",
              nutrients: ["Omega-3", "Vitamin E", "Fiber"]
            }
          ],
          lunch: [
            {
              food: "Salmon Salad with Spinach",
              benefits: "Omega-3 fatty acids reduce inflammation",
              nutrients: ["Omega-3", "Iron", "Vitamin A"]
            },
            {
              food: "Sweet Potato",
              benefits: "Beta-carotene converts to vitamin A for skin repair",
              nutrients: ["Beta-carotene", "Vitamin C", "Fiber"]
            }
          ],
          dinner: [
            {
              food: "Grilled Chicken with Broccoli",
              benefits: "Lean protein for collagen production",
              nutrients: ["Protein", "Vitamin C", "Folate"]
            },
            {
              food: "Quinoa",
              benefits: "Complete protein and B vitamins for skin health",
              nutrients: ["Protein", "B Vitamins", "Magnesium"]
            }
          ],
          snacks: [
            {
              food: "Handful of Walnuts",
              benefits: "Omega-3 fatty acids for skin inflammation",
              nutrients: ["Omega-3", "Vitamin E", "Magnesium"]
            },
            {
              food: "Green Tea",
              benefits: "Antioxidants fight free radical damage",
              nutrients: ["Polyphenols", "EGCG", "Antioxidants"]
            }
          ]
        },
        supplements: [
          {
            name: "Omega-3 Fish Oil",
            dosage: "1000mg daily",
            benefits: "Reduces inflammation and supports skin barrier"
          },
          {
            name: "Vitamin D3",
            dosage: "2000 IU daily",
            benefits: "Supports skin cell growth and repair"
          },
          {
            name: "Zinc",
            dosage: "15mg daily",
            benefits: "Helps with acne and wound healing"
          }
        ],
        hydration: {
          waterGoal: "8-10 glasses daily",
          tips: [
            "Start each day with a glass of water",
            "Add lemon for vitamin C boost",
            "Herbal teas count toward daily intake",
            "Monitor urine color - pale yellow is ideal"
          ]
        },
        avoid: [
          "Excessive dairy (may trigger inflammation)",
          "High glycemic foods (white bread, sugary snacks)",
          "Processed foods high in trans fats",
          "Excessive alcohol consumption"
        ],
        skinFoods: [
          "Tomatoes - lycopene for sun protection",
          "Dark leafy greens - vitamins A, C, E",
          "Citrus fruits - vitamin C for collagen",
          "Nuts and seeds - healthy fats and vitamin E"
        ]
      };
      
      setDietPlan(mockDietPlan);
      setIsGenerating(false);
    }, 2000);
  };

  useEffect(() => {
    if (latestAnalysis && !routine) {
      generateRoutine();
    }
  }, [latestAnalysis, routine, generateRoutine]);

  if (!latestAnalysis) {
    return (
      <div className="skin-routine">
        <div className="empty-state">
          <div className="empty-icon"><Sparkles size={48} /></div>
          <h2>Ready for Your Personalized Routine?</h2>
          <p>Complete your first skin analysis to get AI-powered skincare and diet recommendations tailored just for you!</p>
          <div className="routine-benefits">
            <div className="benefit"><Droplets size={20} className="benefit-icon" /> Customized product recommendations</div>
            <div className="benefit"><Calendar size={20} className="benefit-icon" /> Step-by-step daily routine</div>
            <div className="benefit"><Apple size={20} className="benefit-icon" /> Skin-nourishing diet plan</div>
            <div className="benefit"><Lightbulb size={20} className="benefit-icon" /> Expert tips and guidance</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="skin-routine">
      <div className="routine-header">
        <h2><Sparkles size={24} className="inline-icon" /> Your Personalized Care Plan</h2>
        <p>AI-powered recommendations based on your latest skin analysis</p>
      </div>

      <div className="tab-selector">
        <button 
          className={`tab-btn ${activeTab === 'routine' ? 'active' : ''}`}
          onClick={() => setActiveTab('routine')}
        >
          <Droplets size={20} className="tab-icon" /> Skincare Routine
        </button>
        <button 
          className={`tab-btn ${activeTab === 'diet' ? 'active' : ''}`}
          onClick={() => setActiveTab('diet')}
        >
          <Apple size={20} className="tab-icon" /> Diet Plan
        </button>
      </div>

      {activeTab === 'routine' && (
        <div className="routine-content">
          {isGenerating && !routine ? (
            <div className="generating-state">
              <div className="loading-animation">
                <div className="loading-circle"></div>
              </div>
              <h3>Creating Your Perfect Routine...</h3>
              <p>Analyzing your skin type and concerns to recommend the best products and steps</p>
            </div>
          ) : routine && (
            <>
              <div className="routine-overview">
                <div className="overview-card">
                  <h3>Routine Summary</h3>
                  <div className="summary-info">
                    <div className="info-item">
                      <span className="info-label">Skin Type:</span>
                      <span className="info-value">{routine.skinType}</span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Main Concerns:</span>
                      <div className="concerns-list">
                        {routine.mainConcerns.map((concern, index) => (
                          <span key={index} className="concern-tag">{concern}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="routine-schedules">
                <div className="schedule-section">
                  <h3><Sunrise size={20} className="inline-icon" /> Morning Routine</h3>
                  <div className="routine-steps">
                    {routine.morning.map((step) => (
                      <div key={step.step} className="routine-step">
                        <div className="step-number">{step.step}</div>
                        <div className="step-content">
                          <h4>{step.product}</h4>
                          <p>{step.description}</p>
                          <div className="step-details">
                            <span className="duration">{step.duration}</span>
                            <div className="ingredients">
                              {step.ingredients.map((ingredient, index) => (
                                <span key={index} className="ingredient">{ingredient}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="schedule-section">
                  <h3><Moon size={20} className="inline-icon" /> Evening Routine</h3>
                  <div className="routine-steps">
                    {routine.evening.map((step) => (
                      <div key={step.step} className="routine-step">
                        <div className="step-number">{step.step}</div>
                        <div className="step-content">
                          <h4>{step.product}</h4>
                          <p>{step.description}</p>
                          <div className="step-details">
                            <span className="duration">{step.duration}</span>
                            <div className="ingredients">
                              {step.ingredients.map((ingredient, index) => (
                                <span key={index} className="ingredient">{ingredient}</span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="schedule-section">
                  <h3><CalendarDays size={20} className="inline-icon" /> Weekly Treatments</h3>
                  <div className="weekly-treatments">
                    {routine.weekly.map((treatment, index) => (
                      <div key={index} className="treatment-item">
                        <div className="treatment-frequency">{treatment.frequency}</div>
                        <div className="treatment-content">
                          <h4>{treatment.product}</h4>
                          <p>{treatment.description}</p>
                          <div className="ingredients">
                            {treatment.ingredients.map((ingredient, idx) => (
                              <span key={idx} className="ingredient">{ingredient}</span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="routine-tips">
                  <h3><Lightbulb size={20} className="inline-icon" /> Pro Tips</h3>
                  <ul>
                    {routine.tips.map((tip, index) => (
                      <li key={index}>{tip}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'diet' && (
        <div className="diet-content">
          {!dietPlan ? (
            <div className="generate-diet">
              <div className="generate-card">
                <h3><Apple size={24} className="inline-icon" /> Generate Your Skin-Nourishing Diet Plan</h3>
                <p>Get personalized nutrition recommendations to support your skin health from the inside out.</p>
                <button 
                  className="generate-btn"
                  onClick={generateDietPlan}
                  disabled={isGenerating}
                >
                  {isGenerating ? 'Generating Plan...' : 'Generate Diet Plan'}
                </button>
              </div>
            </div>
          ) : (
            <div className="diet-plan">
              <div className="diet-overview">
                <h3>Your Personalized Nutrition Plan</h3>
                <p>Targeted nutrition to address: {dietPlan.goals.join(', ')}</p>
              </div>

              <div className="daily-meals">
                <div className="meal-section">
                  <h4><Sunrise size={18} className="inline-icon" /> Breakfast</h4>
                  {dietPlan.dailyPlan.breakfast.map((meal, index) => (
                    <div key={index} className="meal-item">
                      <h5>{meal.food}</h5>
                      <p>{meal.benefits}</p>
                      <div className="nutrients">
                        {meal.nutrients.map((nutrient, idx) => (
                          <span key={idx} className="nutrient">{nutrient}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="meal-section">
                  <h4><Sun size={18} className="inline-icon" /> Lunch</h4>
                  {dietPlan.dailyPlan.lunch.map((meal, index) => (
                    <div key={index} className="meal-item">
                      <h5>{meal.food}</h5>
                      <p>{meal.benefits}</p>
                      <div className="nutrients">
                        {meal.nutrients.map((nutrient, idx) => (
                          <span key={idx} className="nutrient">{nutrient}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="meal-section">
                  <h4><Moon size={18} className="inline-icon" /> Dinner</h4>
                  {dietPlan.dailyPlan.dinner.map((meal, index) => (
                    <div key={index} className="meal-item">
                      <h5>{meal.food}</h5>
                      <p>{meal.benefits}</p>
                      <div className="nutrients">
                        {meal.nutrients.map((nutrient, idx) => (
                          <span key={idx} className="nutrient">{nutrient}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="meal-section">
                  <h4><Apple size={18} className="inline-icon" /> Snacks</h4>
                  {dietPlan.dailyPlan.snacks.map((meal, index) => (
                    <div key={index} className="meal-item">
                      <h5>{meal.food}</h5>
                      <p>{meal.benefits}</p>
                      <div className="nutrients">
                        {meal.nutrients.map((nutrient, idx) => (
                          <span key={idx} className="nutrient">{nutrient}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="additional-info">
                <div className="info-section">
                  <h4><Pill size={18} className="inline-icon" /> Recommended Supplements</h4>
                  <div className="supplements">
                    {dietPlan.supplements.map((supplement, index) => (
                      <div key={index} className="supplement-item">
                        <div className="supplement-name">{supplement.name}</div>
                        <div className="supplement-dosage">{supplement.dosage}</div>
                        <div className="supplement-benefits">{supplement.benefits}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="info-section">
                  <h4><DropletIcon size={18} className="inline-icon" /> Hydration Goals</h4>
                  <div className="hydration-info">
                    <div className="water-goal">Target: {dietPlan.hydration.waterGoal}</div>
                    <ul className="hydration-tips">
                      {dietPlan.hydration.tips.map((tip, index) => (
                        <li key={index}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div className="info-section">
                  <h4><X size={18} className="inline-icon" /> Foods to Limit</h4>
                  <ul className="avoid-list">
                    {dietPlan.avoid.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="info-section">
                  <h4><Star size={18} className="inline-icon" /> Skin Superfoods</h4>
                  <div className="superfoods">
                    {dietPlan.skinFoods.map((food, index) => (
                      <div key={index} className="superfood-item">{food}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SkinRoutine;
