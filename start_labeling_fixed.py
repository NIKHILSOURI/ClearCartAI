"""
Start fixed labeling tool
Bypass Gradio 6.x network check issues
"""
import os
import sys

# Set environment variables
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['PYTHONPATH'] = r'D:\Download\ean_system_repo\ean_system'

# Add to Python path
sys.path.insert(0, r'D:\Download\ean_system_repo\ean_system')

print('='*60)
print('🚀 Starting SAM3 Product Labeling Tool (Fixed Version)')
print('='*60)
print('')
print('📍 Configuration:')
print('   - Image directory: raw_pictures\\')
print('   - Access address: http://127.0.0.1:7860')
print('')
print('⏳ Starting server...')
print('')

# Import labeling tool
from tools.label_ui_gradio import demo

print('='*60)
print('✅ Labeling tool started!')
print('📍 Open in browser: http://127.0.0.1:7860')
print('='*60)
print('')
print('💡 Usage Instructions:')
print('   1. Click "Load Next" to load image')
print('   2. Click on product center')
print('   3. Click "Run SAM 3" (wait 8-10 seconds)')
print('   4. Select packaging type, enter product name')
print('   5. Click "Save Record" to save and load next')
print('')
print('⚠️  Note: SAM processing takes 8-10 seconds, please wait for status updates')
print('='*60)
print('')

# Use FastAPI directly to bypass Gradio network check
try:
    import uvicorn
    from gradio.routes import App
    
    # Create FastAPI app
    app = App.create_app(demo)
    
    # Run uvicorn directly
    uvicorn.run(
        app, 
        host='127.0.0.1', 
        port=7860,
        log_level='warning',
        access_log=False
    )
except KeyboardInterrupt:
    print('\n\n👋 Labeling tool stopped')
except Exception as e:
    print(f'\n❌ Error: {e}')
    import traceback
    traceback.print_exc()
