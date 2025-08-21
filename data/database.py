"""
Database models for Hospital Procurement System
SQLAlchemy models for purchase orders, vendors, and related data
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import pandas as pd
from typing import Optional, List

Base = declarative_base()


class Vendor(Base):
    """Vendor information table"""
    __tablename__ = 'vendors'
    
    vendor_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    contact_person = Column(String(255))
    email = Column(String(255))
    phone = Column(String(50))
    address = Column(Text)
    category = Column(String(100))
    rating = Column(Float, default=0.0)
    status = Column(String(50), default='active')  # active, inactive, suspended
    contract_start_date = Column(DateTime)
    contract_end_date = Column(DateTime)
    payment_terms = Column(String(100))
    delivery_lead_time = Column(Integer)  # in days
    quality_score = Column(Float, default=0.0)
    reliability_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    purchase_orders = relationship("PurchaseOrder", back_populates="vendor")
    items = relationship("Item", back_populates="vendor")


class Item(Base):
    """Item catalog table"""
    __tablename__ = 'items'
    
    item_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    subcategory = Column(String(100))
    unit = Column(String(50))  # pieces, kg, liters, etc.
    vendor_id = Column(Integer, ForeignKey('vendors.vendor_id'))
    unit_price = Column(Float)
    min_order_quantity = Column(Integer)
    lead_time = Column(Integer)  # in days
    is_critical = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    vendor = relationship("Vendor", back_populates="items")
    order_items = relationship("OrderItem", back_populates="item")


class PurchaseOrder(Base):
    """Purchase order table"""
    __tablename__ = 'purchase_orders'
    
    order_id = Column(Integer, primary_key=True)
    order_number = Column(String(100), unique=True, nullable=False)
    vendor_id = Column(Integer, ForeignKey('vendors.vendor_id'))
    order_date = Column(DateTime, nullable=False)
    expected_delivery_date = Column(DateTime)
    actual_delivery_date = Column(DateTime)
    status = Column(String(50), default='pending')  # pending, approved, delivered, cancelled
    total_amount = Column(Float, default=0.0)
    currency = Column(String(10), default='USD')
    payment_terms = Column(String(100))
    notes = Column(Text)
    created_by = Column(String(100))
    approved_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    vendor = relationship("Vendor", back_populates="purchase_orders")
    order_items = relationship("OrderItem", back_populates="purchase_order")


class OrderItem(Base):
    """Purchase order items table"""
    __tablename__ = 'order_items'
    
    order_item_id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('purchase_orders.order_id'))
    item_id = Column(Integer, ForeignKey('items.item_id'))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    received_quantity = Column(Integer, default=0)
    quality_rating = Column(Float)
    notes = Column(Text)
    
    # Relationships
    purchase_order = relationship("PurchaseOrder", back_populates="order_items")
    item = relationship("Item", back_populates="order_items")


class ProcurementDatabase:
    """Database manager for procurement system"""
    
    def __init__(self, database_url: str = "sqlite:///procurement.db"):
        """
        Initialize database connection
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def get_purchase_orders_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get purchase orders as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(PurchaseOrder)
            if limit:
                query = query.limit(limit)
            
            # Convert to DataFrame
            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()
    
    def get_vendors_df(self) -> pd.DataFrame:
        """Get vendors as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(Vendor)
            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()
    
    def get_items_df(self) -> pd.DataFrame:
        """Get items as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(Item)
            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()
    
    def get_order_items_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get order items as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(OrderItem)
            if limit:
                query = query.limit(limit)
            
            df = pd.read_sql(query.statement, session.bind)
            return df
        finally:
            session.close()
    
    def get_procurement_summary(self) -> dict:
        """Get summary statistics of procurement data"""
        session = self.get_session()
        try:
            # Count records
            vendor_count = session.query(Vendor).count()
            order_count = session.query(PurchaseOrder).count()
            item_count = session.query(Item).count()
            
            # Total spend
            total_spend = session.query(PurchaseOrder.total_amount).filter(
                PurchaseOrder.status == 'delivered'
            ).scalar() or 0.0
            
            # Average order value
            avg_order_value = session.query(PurchaseOrder.total_amount).filter(
                PurchaseOrder.status == 'delivered'
            ).scalar() or 0.0
            
            return {
                "vendor_count": vendor_count,
                "order_count": order_count,
                "item_count": item_count,
                "total_spend": total_spend,
                "avg_order_value": avg_order_value
            }
        finally:
            session.close()
    
    def add_sample_data(self):
        """Add sample data for testing"""
        session = self.get_session()
        try:
            # Add vendors
            vendors = [
                Vendor(
                    name="Medical Supplies Co.",
                    contact_person="John Smith",
                    email="john@medsupplies.com",
                    phone="555-0101",
                    category="Medical Supplies",
                    rating=4.5,
                    status="active",
                    quality_score=4.2,
                    reliability_score=4.8
                ),
                Vendor(
                    name="Hospital Equipment Inc.",
                    contact_person="Sarah Johnson",
                    email="sarah@hospitequip.com",
                    phone="555-0102",
                    category="Equipment",
                    rating=4.2,
                    status="active",
                    quality_score=4.0,
                    reliability_score=4.5
                ),
                Vendor(
                    name="Pharmaceutical Solutions",
                    contact_person="Mike Brown",
                    email="mike@pharmsol.com",
                    phone="555-0103",
                    category="Pharmaceuticals",
                    rating=4.8,
                    status="active",
                    quality_score=4.7,
                    reliability_score=4.9
                )
            ]
            
            for vendor in vendors:
                session.add(vendor)
            
            session.commit()
            
            # Add items
            items = [
                Item(
                    name="Surgical Masks",
                    description="Disposable surgical face masks",
                    category="PPE",
                    subcategory="Face Protection",
                    unit="box",
                    unit_price=25.0,
                    min_order_quantity=10,
                    lead_time=7,
                    is_critical=True
                ),
                Item(
                    name="Syringes 10ml",
                    description="Disposable syringes 10ml",
                    category="Medical Supplies",
                    subcategory="Injection",
                    unit="piece",
                    unit_price=0.50,
                    min_order_quantity=100,
                    lead_time=5,
                    is_critical=True
                ),
                Item(
                    name="Hospital Beds",
                    description="Electric hospital beds",
                    category="Equipment",
                    subcategory="Furniture",
                    unit="piece",
                    unit_price=2500.0,
                    min_order_quantity=1,
                    lead_time=14,
                    is_critical=False
                )
            ]
            
            for item in items:
                session.add(item)
            
            session.commit()
            
            # Add purchase orders
            orders = [
                PurchaseOrder(
                    order_number="PO-2024-001",
                    vendor_id=1,
                    order_date=datetime(2024, 1, 15),
                    expected_delivery_date=datetime(2024, 1, 22),
                    status="delivered",
                    total_amount=2500.0,
                    created_by="procurement@hospital.com"
                ),
                PurchaseOrder(
                    order_number="PO-2024-002",
                    vendor_id=2,
                    order_date=datetime(2024, 1, 20),
                    expected_delivery_date=datetime(2024, 2, 3),
                    status="approved",
                    total_amount=5000.0,
                    created_by="procurement@hospital.com"
                ),
                PurchaseOrder(
                    order_number="PO-2024-003",
                    vendor_id=3,
                    order_date=datetime(2024, 1, 25),
                    expected_delivery_date=datetime(2024, 1, 30),
                    status="pending",
                    total_amount=1500.0,
                    created_by="procurement@hospital.com"
                )
            ]
            
            for order in orders:
                session.add(order)
            
            session.commit()
            
            print("Sample data added successfully!")
            
        except Exception as e:
            session.rollback()
            print(f"Error adding sample data: {e}")
        finally:
            session.close()
