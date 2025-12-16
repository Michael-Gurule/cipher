"""
Generate synthetic marketing attribution data with known causal effects
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


class MarketingDataGenerator:
    """Generate realistic marketing attribution data with causal effects"""
    
    def __init__(self, n_users=100000, date_range_days=365):
        self.n_users = n_users
        self.date_range_days = date_range_days
        self.channels = ['Google_Ads', 'Facebook', 'Email', 'Organic', 'Referral']
        self.start_date = datetime(2024, 1, 1)
        
        # True causal effects (ground truth)
        self.true_effects = {
            'Google_Ads': {
                'base_lift': 0.12,  # 12% lift on average
                'by_segment': {'New': 0.15, 'Returning': 0.10, 'VIP': 0.08}
            },
            'Facebook': {
                'base_lift': 0.08,
                'by_segment': {'New': 0.12, 'Returning': 0.06, 'VIP': 0.04}
            },
            'Email': {
                'base_lift': 0.18,  # Email is very effective
                'by_segment': {'New': -0.05, 'Returning': 0.25, 'VIP': 0.30}  # Negative for new users!
            },
            'Organic': {
                'base_lift': 0.15,
                'by_segment': {'New': 0.18, 'Returning': 0.14, 'VIP': 0.12}
            },
            'Referral': {
                'base_lift': 0.20,  # Referrals are high quality
                'by_segment': {'New': 0.22, 'Returning': 0.18, 'VIP': 0.16}
            }
        }
    
    def generate_users(self):
        """Generate user profiles with demographics"""
        users = []
        
        for user_id in range(self.n_users):
            # Demographics
            age = int(np.random.normal(38, 12))
            age = max(18, min(75, age))  # Clip to reasonable range
            
            gender = np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04])
            
            location = np.random.choice([
                'California', 'Texas', 'Florida', 'New York', 'Illinois',
                'Pennsylvania', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'
            ])
            
            # Customer segment
            segment_probs = [0.45, 0.40, 0.15]  # New, Returning, VIP
            segment = np.random.choice(['New', 'Returning', 'VIP'], p=segment_probs)
            
            # Intent score (natural propensity to convert)
            if segment == 'VIP':
                intent_score = np.random.beta(8, 2)  # High intent
            elif segment == 'Returning':
                intent_score = np.random.beta(4, 4)  # Medium intent
            else:
                intent_score = np.random.beta(2, 5)  # Lower intent
            
            users.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'location': location,
                'segment': segment,
                'intent_score': round(intent_score, 3)
            })
        
        return pd.DataFrame(users)
    
    def generate_channel_exposure(self, user_df):
        """Determine which channels each user is exposed to (with selection bias)"""
        exposures = []
        
        for _, user in user_df.iterrows():
            # Channel exposure probabilities depend on user characteristics (selection bias!)
            
            # High-intent users more likely to search (see Google)
            google_prob = 0.3 + (user['intent_score'] * 0.4)
            
            # Younger users more likely on social media
            age_factor = max(0, (45 - user['age']) / 45)
            facebook_prob = 0.2 + (age_factor * 0.3)
            
            # Returning customers more likely to get emails
            email_prob = 0.4 if user['segment'] in ['Returning', 'VIP'] else 0.1
            
            # Everyone can see organic (but high-intent users search more)
            organic_prob = 0.5 + (user['intent_score'] * 0.3)
            
            # Referrals are random but higher for existing customers
            referral_prob = 0.15 if user['segment'] != 'New' else 0.05
            
            # Determine exposures
            exposed_channels = []
            if np.random.random() < google_prob:
                exposed_channels.append('Google_Ads')
            if np.random.random() < facebook_prob:
                exposed_channels.append('Facebook')
            if np.random.random() < email_prob:
                exposed_channels.append('Email')
            if np.random.random() < organic_prob:
                exposed_channels.append('Organic')
            if np.random.random() < referral_prob:
                exposed_channels.append('Referral')
            
            # Ensure at least one touchpoint
            if not exposed_channels:
                exposed_channels = [np.random.choice(self.channels)]
            
            exposures.append({
                'user_id': user['user_id'],
                'exposed_channels': exposed_channels,
                'n_touchpoints': len(exposed_channels)
            })
        
        return pd.DataFrame(exposures)
    
    def generate_conversions(self, user_df, exposure_df):
        """Generate conversion outcomes with true causal effects"""
        results = []
        
        merged = user_df.merge(exposure_df, on='user_id')
        
        for _, row in merged.iterrows():
            # Base conversion probability (without any marketing)
            base_prob = row['intent_score'] * 0.15  # Max 15% natural conversion
            
            # Add causal effects from each channel
            total_lift = 0
            channel_effects = {}
            
            for channel in row['exposed_channels']:
                # Get true effect for this channel and segment
                effect = self.true_effects[channel]['by_segment'][row['segment']]
                total_lift += effect
                channel_effects[channel] = effect
            
            # Diminishing returns if too many touchpoints
            if row['n_touchpoints'] > 3:
                total_lift *= 0.85  # Fatigue effect
            
            # Final conversion probability
            conversion_prob = base_prob + total_lift
            conversion_prob = max(0, min(1, conversion_prob))  # Clip to [0,1]
            
            # Convert
            converted = np.random.random() < conversion_prob
            
            # Revenue if converted
            if converted:
                # Revenue depends on segment
                if row['segment'] == 'VIP':
                    revenue = np.random.lognormal(5.0, 0.5)  # High value
                elif row['segment'] == 'Returning':
                    revenue = np.random.lognormal(4.2, 0.6)  # Medium value
                else:
                    revenue = np.random.lognormal(3.5, 0.7)  # Lower value
                revenue = min(500, revenue)  # Cap at $500
            else:
                revenue = 0
            
            # Create customer journey
            journey = []
            for i, channel in enumerate(row['exposed_channels']):
                days_before = random.randint(0, 7)  # Within last week
                hours = random.randint(0, 23)
                minutes = random.randint(0, 59)
                
                timestamp = self.start_date + timedelta(
                    days=random.randint(0, self.date_range_days-7) + days_before,
                    hours=hours,
                    minutes=minutes
                )
                
                journey.append({
                    'channel': channel,
                    'timestamp': timestamp.isoformat(),
                    'position': i + 1
                })
            
            # Sort journey by timestamp
            journey = sorted(journey, key=lambda x: x['timestamp'])
            
            results.append({
                'user_id': row['user_id'],
                'journey': json.dumps(journey),
                'n_touchpoints': row['n_touchpoints'],
                'converted': int(converted),
                'revenue': round(revenue, 2),
                'base_probability': round(base_prob, 4),
                'total_lift': round(total_lift, 4),
                'final_probability': round(conversion_prob, 4),
                'attribution_truth': json.dumps(channel_effects)
            })
        
        return pd.DataFrame(results)
    
    def generate_channel_metrics(self):
        """Generate daily channel-level metrics"""
        dates = [self.start_date + timedelta(days=x) for x in range(self.date_range_days)]
        
        metrics = []
        
        for date in dates:
            # Day of week effects (weekends different)
            dow = date.weekday()
            weekend_factor = 0.7 if dow >= 5 else 1.0
            
            # Seasonal trend (Q4 higher)
            month = date.month
            seasonal_factor = 1.2 if month in [11, 12] else 1.0
            
            for channel in self.channels:
                # Base daily spend
                if channel == 'Google_Ads':
                    base_spend = 11000
                    base_impressions = 500000
                elif channel == 'Facebook':
                    base_spend = 8200
                    base_impressions = 800000
                elif channel == 'Email':
                    base_spend = 4100
                    base_impressions = 200000
                elif channel == 'Organic':
                    base_spend = 2700  # SEO costs
                    base_impressions = 300000
                else:  # Referral
                    base_spend = 1400
                    base_impressions = 50000
                
                # Add noise
                spend = base_spend * weekend_factor * seasonal_factor * np.random.uniform(0.9, 1.1)
                impressions = int(base_impressions * weekend_factor * seasonal_factor * np.random.uniform(0.85, 1.15))
                
                # Clicks (CTR varies by channel)
                ctr = {
                    'Google_Ads': 0.025,
                    'Facebook': 0.015,
                    'Email': 0.08,
                    'Organic': 0.05,
                    'Referral': 0.20
                }[channel]
                
                clicks = int(impressions * ctr * np.random.uniform(0.9, 1.1))
                
                metrics.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'channel': channel,
                    'spend': round(spend, 2),
                    'impressions': impressions,
                    'clicks': clicks,
                })
        
        return pd.DataFrame(metrics)
    
    def generate_complete_dataset(self):
        """Generate all datasets"""
        print("Generating synthetic marketing attribution data...")
        print(f"Creating {self.n_users:,} user profiles...")
        
        # Generate users
        user_df = self.generate_users()
        print(f"✓ Generated {len(user_df):,} users")
        
        # Generate channel exposures
        print("Determining channel exposures...")
        exposure_df = self.generate_channel_exposure(user_df)
        print(f"✓ Generated exposures (avg {exposure_df['n_touchpoints'].mean():.1f} touchpoints/user)")
        
        # Generate conversions
        print("Simulating conversions with causal effects...")
        conversion_df = self.generate_conversions(user_df, exposure_df)
        print(f"✓ Generated conversions ({conversion_df['converted'].sum():,} conversions, {conversion_df['converted'].mean()*100:.1f}% rate)")
        
        # Merge all data
        complete_df = user_df.merge(conversion_df, on='user_id')
        
        # Generate channel metrics
        print("Creating daily channel metrics...")
        channel_metrics_df = self.generate_channel_metrics()
        print(f"✓ Generated {len(channel_metrics_df):,} daily channel records")
        
        return complete_df, channel_metrics_df
    
    def save_data(self, user_journey_df, channel_metrics_df):
        """Save datasets to CSV"""
        import os
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save user journey data
        user_journey_path = 'data/marketing_journeys.csv'
        user_journey_df.to_csv(user_journey_path, index=False)
        print(f"\n✓ Saved user journeys to: {user_journey_path}")
        print(f"  Shape: {user_journey_df.shape}")
        print(f"  Size: ~{os.path.getsize(user_journey_path) / 1024 / 1024:.1f} MB")
        
        # Save channel metrics
        channel_path = 'data/channel_metrics.csv'
        channel_metrics_df.to_csv(channel_path, index=False)
        print(f"\n✓ Saved channel metrics to: {channel_path}")
        print(f"  Shape: {channel_metrics_df.shape}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        print("\nUser Journey Data:")
        print(f"  Total users: {len(user_journey_df):,}")
        print(f"  Conversions: {user_journey_df['converted'].sum():,} ({user_journey_df['converted'].mean()*100:.1f}%)")
        print(f"  Total revenue: ${user_journey_df['revenue'].sum():,.0f}")
        print(f"  Avg revenue per conversion: ${user_journey_df[user_journey_df['converted']==1]['revenue'].mean():.2f}")
        
        print("\nBy Segment:")
        for segment in ['New', 'Returning', 'VIP']:
            segment_data = user_journey_df[user_journey_df['segment'] == segment]
            conv_rate = segment_data['converted'].mean() * 100
            print(f"  {segment:10s}: {len(segment_data):6,} users, {conv_rate:5.2f}% conversion")
        
        print("\nChannel Metrics:")
        print(f"  Date range: {channel_metrics_df['date'].min()} to {channel_metrics_df['date'].max()}")
        print(f"  Total spend: ${channel_metrics_df['spend'].sum():,.0f}")
        print(f"  By channel:")
        for channel in self.channels:
            channel_data = channel_metrics_df[channel_metrics_df['channel'] == channel]
            total_spend = channel_data['spend'].sum()
            print(f"    {channel:12s}: ${total_spend:>10,.0f}")
        
        print("\n" + "="*60)
        print("Data generation complete!")
        print("="*60)


if __name__ == "__main__":
    # Generate data
    generator = MarketingDataGenerator(n_users=100000, date_range_days=365)
    user_journey_df, channel_metrics_df = generator.generate_complete_dataset()
    
    # Save to CSV
    generator.save_data(user_journey_df, channel_metrics_df)
    
    print("\n✅ Ready for analysis!")
    print("\nNext steps:")
    print("  1. Run exploratory data analysis")
    print("  2. Start with notebooks/01_propensity_score_matching.ipynb")